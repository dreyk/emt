import tensorflow as tf
import logging
import glob
import os


def data_fn(args, training):
    data_set = args.data_set
    files = glob.glob(data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = data_set + '/images/' + img
        files[i] = [img, mask]
    resolution = args.resolution
    logging.info('Number of Files: {}'.format(len(files)))

    ds = tf.data.Dataset.from_tensor_slices(files)

    def _read_images(a):
        img = tf.io.read_file(a[0])
        img = tf.io.decode_image(img,expand_animations = False)
        mask = tf.io.read_file(a[1])
        mask = tf.io.decode_image(mask,expand_animations = False)
        img = tf.expand_dims(img, 0)
        mask = tf.expand_dims(mask, 0)
        img = tf.image.resize(img, [resolution, resolution])
        mask = tf.image.resize(mask, [resolution, resolution])
        logging.info('img: {}'.format(img.shape))
        logging.info('mask: {}'.format(mask.shape))
        img = tf.reshape(img, [resolution, resolution, 3])
        mask = tf.reshape(mask[:, :, :, 0], [resolution, resolution, 1])
        img = tf.cast(img, dtype=tf.float32) / 255
        mask = tf.cast(mask, dtype=tf.float32) / 255
        weight = tf.ones_like(mask, dtype=tf.float32)
        return img, tf.concat([mask, weight], axis=2)

    ds = ds.map(_read_images)
    if training:
        ds = ds.shuffle(args.batch_size * 2, reshuffle_each_iteration=True)
    ds = ds.batch(args.batch_size, True)

    return ds
