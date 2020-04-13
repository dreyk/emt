import tensorflow as tf
import tensorflow.keras as keras
import data.alpha_base as alpha_base
import models.fc_densenet.model as fc_densenet
import logging
import os
import argparse
import shutil



def train(args):
    logdir = args.checkpoint_dir
    os.makedirs(logdir)
    file_writer = tf.summary.create_file_writer(logdir)
    #images_file_writer = tf.summary.create_file_writer(logdir + "/images")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    ds = alpha_base.data_fn(args,True)
    model = fc_densenet.FCDensNet(input_shape=(160,160,3))
    #model.compile(
    #    loss=tf.keras.losses.MeanAbsoluteError(),  # keras.losses.mean_squared_error
    #    optimizer=keras.optimizers.Adam(),
    #)
    model.summary()

    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam()
    for step, (x_batch_train, y_batch_train) in enumerate(ds):
        with tf.GradientTape() as tape:
            outputs = model(x_batch_train, training=True)  # Logits for this minibatch
            alpha = outputs[0]
            loss_value = loss_fn(y_batch_train, alpha)
            if step % 50 == 0:
                logging.info("Step {}: Loss={}".format(step,loss_value))
                with file_writer.as_default():
                    tf.summary.scalar("Loss",loss_value,step=step)
                    tf.summary.image("Src", x_batch_train, step=step,max_outputs=3)
                    tf.summary.image("Original", y_batch_train, step=step, max_outputs=3)
                    tf.summary.image("Results", alpha, step=step, max_outputs=3)
                    tf.summary.image("Features-0", outputs[1][:,:,:,0:1], step=step, max_outputs=1)
                    tf.summary.image("Features-1", outputs[1][:, :, :, 1:2], step=step, max_outputs=1)
                    tf.summary.image("Features-3", outputs[1][:, :, :, 2:3], step=step, max_outputs=1)
                    for i in range(len(outputs)-2):
                        tf.summary.image(f"Feature0-{i}",outputs[i+2][:,:,:,0:1], step=step, max_outputs=1)
                        tf.summary.image(f"Feature2-{i}", outputs[i + 2][:, :, :, 1:2], step=step, max_outputs=1)
                        tf.summary.image(f"Feature3-{i}", outputs[i + 2][:, :, :, 2:3], step=step, max_outputs=1)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


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
    parser.add_argument('--batch-size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--resolution', default=160, type=int, help='Resolution of images')
    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')

    return parser
if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    train(args)