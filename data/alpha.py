import tensorflow as tf
import os
import cv2
import numpy as np
import glob


def data_fn(args, training):
    files = glob.glob(args.data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = args.data_set + '/images/' + img
        files[i] = (img, mask)

    def _generator():

        for i in files:
            img = cv2.imread(i[0])
            mask = cv2.imread(i[1])
            img = cv2.resize(img,(args.resolution,args.resolution))
            mask = cv2.resize(mask, (args.resolution, args.resolution))
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            img = img.astype(np.float32)/255
            mask = mask.astype(np.float32)/255
            mask = np.expand_dims(mask,axis=2)
            bg = img*(1-mask)
            fg = img*mask
            yield img, np.concatenate([mask, fg, bg], axis=2)

    ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                        (tf.TensorShape([args.resolution, args.resolution, 3]),
                                         tf.TensorShape([args.resolution, args.resolution, 7])))
    if training:
        ds = ds.shuffle(args.batch_size * 3, reshuffle_each_iteration=True)

    ds = ds.batch(args.batch_size, True)

    return ds
