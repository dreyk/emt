import tensorflow as tf
import data.alpha_base as data
import models.fc_densenet.matting as fc_densenet
import logging
import os
import argparse
import numpy as np




def gauss_kernel(size=5, sigma=1.0):
  grid = np.float32(np.mgrid[0:size,0:size].T)
  gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
  kernel = np.sum(gaussian(grid), axis=2)
  kernel /= np.sum(kernel)
  return kernel

def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
  t_kernel = tf.reshape(tf.constant(gauss_kernel(size=k_size, sigma=sigma), tf.float32),
                        [k_size, k_size, 1, 1])
  t_kernel3 = tf.concat([t_kernel]*t_input.get_shape()[3], axis=2)
  t_result = t_input
  for r in range(repeats):
    t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3,
        strides=[1, stride, stride, 1], padding='SAME')
  return t_result

def make_laplacian_pyramid(t_img, max_levels):
  t_pyr = []
  current = t_img
  for level in range(max_levels):
    t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
    t_diff = current - t_gauss
    t_pyr.append(t_diff)
    current = tf.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
  t_pyr.append(current)
  return t_pyr

def laploss(t_img1, t_img2, max_levels=3):
  t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
  t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
  t_losses = [tf.norm(a-b,ord=1)/tf.size(a, out_type=tf.float32) for a,b in zip(t_pyr1, t_pyr2)]
  t_loss = tf.reduce_sum(t_losses)
  return t_loss



def train(args):
    logdir = args.checkpoint_dir
    os.makedirs(logdir)
    file_writer = tf.summary.create_file_writer(logdir)
    ds = data.augumnted_data_fn(args, True)
    model = fc_densenet.FCDensNetMatting()
    model.summary()

    l1 = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    step = 0
    for e in range(args.num_epochs):
        for (x_batch_train, y_batch_train) in ds:
            with tf.GradientTape() as tape:
                img = x_batch_train[:,:,:,0:3]
                trimap = x_batch_train[:,:,:,3:]
                outputs = model([img,trimap], training=True)  # Logits for this minibatch
                palpha = outputs[0]
                palpha_dx, palpha_dy = tf.image.image_gradients(palpha)
                pfg = outputs[1]
                pbg = outputs[2]
                alpha = y_batch_train[:,:,:,0:1]
                alpha_dx, alpha_dy = tf.image.image_gradients(alpha)
                fg = y_batch_train[:, :, :, 1:4]
                bg = y_batch_train[:, :, :, 4:7]
                alpha_l1 = l1(alpha,palpha)
                #alpha_g = l1(alpha_dx,palpha_dx)+l1(alpha_dy,palpha_dy)
                alpha_c = l1(img, fg * palpha + bg * (1 - palpha))
                alpha_lap = laploss(alpha,palpha)

                fb_l1 = l1(fg,pfg)+l1(bg, pbg)
                pfg_dx,pfg_dy = tf.image.image_gradients(pfg)
                pbg_dx, pbg_dy = tf.image.image_gradients(pbg)
                fb_exl = tf.reduce_mean(tf.abs(pfg_dx)*tf.abs(pbg_dx)/4+tf.abs(pfg_dy)*tf.abs(pbg_dy)/4)
                fb_c = l1(img,alpha*pfg+(1-alpha)*pbg)
                fb_lap = laploss(fg,pfg)+laploss(bg,pbg)

                loss_value = alpha_l1+alpha_c+alpha_lap+0.25*(fb_l1+fb_exl+fb_c+fb_lap)


                if step % 50 == 0:
                    logging.info("Step {}: Loss={}".format(step, loss_value))
                    with file_writer.as_default():
                        tf.summary.scalar("Loss", loss_value, step=step)
                        tf.summary.scalar("Alpha/L1", alpha_l1, step=step)
                        #tf.summary.scalar("Alpha/G", alpha_g, step=step)
                        tf.summary.scalar("Alpha/C", alpha_c, step=step)
                        tf.summary.scalar("Alpha/Lap", alpha_lap, step=step)
                        tf.summary.scalar("FB/L1", fb_l1, step=step)
                        tf.summary.scalar("FB/Exl", fb_exl, step=step)
                        tf.summary.scalar("FB/Lap", fb_lap, step=step)

                        tf.summary.image("Src", img, step=step, max_outputs=3)
                        tf.summary.image("BG", bg, step=step, max_outputs=3)
                        tf.summary.image("FG", fg, step=step, max_outputs=3)
                        tf.summary.image("PBG", pbg, step=step, max_outputs=3)
                        tf.summary.image("PFG", pfg, step=step, max_outputs=3)
                        tf.summary.image("Alpha", alpha, step=step, max_outputs=3)
                        tf.summary.image("Trimap", trimap, step=step, max_outputs=3)
                        tf.summary.image("PAlpha", palpha, step=step, max_outputs=3)
                        tf.summary.image("Res", img*palpha, step=step, max_outputs=3)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            step += 1


def create_arg_parser():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs')
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.info('Checkpoint %s',checkpoint_dir)
    parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--coco', type=str, required=True,
                        help='Coco Path')
    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')

    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()

    train(args)
