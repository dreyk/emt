import tensorflow as tf

import models.unet.unet as unet
import data.one_person as data
import logging
import os
import argparse




def train(args):
    logdir = args.checkpoint_dir
    os.makedirs(logdir)
    file_writer = tf.summary.create_file_writer(logdir)
    ds = data.data_fn(args, True)
    model = unet.unet((args.resolution, args.resolution, 3), first_chan=32, pools=4, growth_add=8, growth_scale=0,
                      out_chans=1, use_group_norm=args.batch_size == 1)
    model.summary()
    l1 = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    step = 0
    tf.summary.trace_on(graph=True, profiler=False)
    for e in range(args.num_epochs):
        logging.info('epoch %d',e)
        for (img, y_batch_train) in ds:
            with tf.GradientTape() as tape:
                outputs = model(img, training=True)  # Logits for this minibatch
                palpha = outputs
                alpha = y_batch_train[:, :, :, 0:1]
                alpha_l1 = l1(alpha, palpha)

                loss_value = alpha_l1

                if step % 10 == 0:
                    logging.info("Step {}: Loss={}".format(step, loss_value))
                    model.save(os.path.join(logdir, 'model'), save_format='tf')
                    with file_writer.as_default():
                        if step==0:
                            tf.summary.trace_export('grpah',0)
                            tf.summary.trace_off()
                        tf.summary.scalar("Loss", loss_value, step=step)
                        tf.summary.scalar("Alpha/L1", alpha_l1, step=step)
                        tf.summary.image("Src", img, step=step, max_outputs=3)
                        tf.summary.image("Alpha", alpha, step=step, max_outputs=3)
                        tf.summary.image("PAlpha", palpha, step=step, max_outputs=3)
                        tf.summary.image("Res", img * palpha, step=step, max_outputs=3)

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
    logging.info('Checkpoint %s', checkpoint_dir)
    parser.add_argument('--batch_size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--coco', default='test', type=str, help='Coco path')
    parser.add_argument('--epoch_len', default=10, type=int, help='Repeat at least')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--resolution', default=160, type=int, help='Resolution of images')
    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')

    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()

    train(args)
