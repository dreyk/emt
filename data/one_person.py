import tensorflow as tf
import os
import cv2
import numpy as np
import glob
import data.coco as coco
import logging

def data_fn(args, training):
    files = glob.glob(args.data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = args.data_set + '/images/' + img
        files[i] = (img, mask)
    logging.info('Number of training files: {}'.format(len(files)))
    coco_bg = coco.CocoBG(args.coco)
    def _generator():
        for _ in range(args.epoch_len):
            for i in files:
                img = cv2.imread(i[0])[:,:,::-1]
                mask = cv2.imread(i[1])
                img = cv2.resize(img,(args.resolution,args.resolution))
                mask = cv2.resize(mask, (args.resolution, args.resolution))
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                bg = coco_bg.get_random(args.resolution,args.resolution)
                bg = bg.astype(np.float32)
                img = img.astype(np.float32)
                mask = mask.astype(np.float32)/255
                mask = np.expand_dims(mask,axis=2)
                img = img*mask+bg*(1-mask)
                yield img, mask

    ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                        (tf.TensorShape([args.resolution, args.resolution, 3]),
                                         tf.TensorShape([args.resolution, args.resolution, 1])))
    if training:
        ds = ds.shuffle(args.batch_size * 3, reshuffle_each_iteration=True)

    ds = ds.batch(args.batch_size, True)

    return ds
