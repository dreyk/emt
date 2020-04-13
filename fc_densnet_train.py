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
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    ds = alpha_base.data_fn(args,True)
    model = fc_densenet.FCDensNet()
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),  # keras.losses.mean_squared_error
        optimizer=keras.optimizers.Adam(),
    )
    model.summary()
    training_history = model.fit(ds,epochs=args.num_epochs,callbacks=[tensorboard_callback])

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
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--resolution', default=160, type=int, help='Resolution of images')
    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')

    return parser
if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    train(args)