import tensorflow as tf
import logging
import glob
import os
import json
import numpy as np
import cv2
import random

from scipy import ndimage

unknown_code = 128


def pre_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 5
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = unknown_code
    return trimap



def generate_trimap(alpha):
    trimap = pre_trimap(alpha)
    k_size = int(np.random.uniform(1, 5)) * 2 + 1
    trimap = cv2.GaussianBlur(trimap, k_size)
    trimap = trimap.astype(np.float32) / 255
    trimap = np.expand_dims(trimap,axis=2)
    return trimap


def random_choice(img,mask,resolution):
    crop_size = random.choice([resolution,resolution+resolution//2,resolution*2])
    trimap = pre_trimap(mask)
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x0 = max(0, center_x - int(crop_size / 2))
        y0 = max(0, center_y - int(crop_size / 2))
        x1 = min(x0+crop_size,img.shape[1])
        y1 = min(y0 +crop_size, img.shape[0])
        if (x1-x0)>crop_size/2 and (y1-y0)>crop_size/2:
            img = img[y0:y1,x0:x1,:]
            mask = mask[y0:y1,x0:x1]
    return img,mask

def _coco_bg(args):
    if args.coco == 'test':
        return ['./testdata/default.png']
    with open(args.coco + '/annotations/instances_train2017.json') as f:
        coco_data = json.load(f)
    coco_data = coco_data['annotations']
    coco_images = {}
    people = {}
    for a in coco_data:
        i_id = a['image_id']
        if a['category_id'] != 1:
            if i_id in people:
                continue
            else:
                coco_images[i_id] = True
        else:
            if i_id in coco_images:
                del coco_images[i_id]
            people[i_id] = True
    del people
    names = []
    for k in coco_images.keys():
        name = '{}/train2017/{:012d}.jpg'.format(args.coco, int(k))
        names.append(name)
    return names


def _crop_back(img, w, h):
    nw = max(img.shape[1], w)
    nh = max(img.shape[0], h)
    img = cv2.resize(img, (nw, nh))
    x_shift = int(np.random.uniform(0, nw - w))
    y_shift = int(np.random.uniform(0, nh - h))
    return img[y_shift:y_shift + h, x_shift:x_shift + w, :]


def _resize_and_put(img, x_shift, y_shift, iw, ih, w, h):
    img = cv2.resize(img, (iw, ih))
    if len(img.shape) == 3:
        res = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        res[y_shift:y_shift + ih, x_shift:x_shift + iw, :] = img
    else:
        res = np.zeros((h, w), dtype=img.dtype)
        res[y_shift:y_shift + ih, x_shift:x_shift + iw] = img
    return res


def augumnted_data_fn(args, training):
    import albumentations
    def _strong_aug(p=0.5):
        return albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.3),
            albumentations.OneOf([
                albumentations.OpticalDistortion(p=0.3),
                albumentations.GridDistortion(p=0.1),
                albumentations.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            albumentations.OneOf([
                albumentations.CLAHE(clip_limit=2),
                albumentations.IAASharpen(),
                albumentations.IAAEmboss(),
            ], p=0.3),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(p=0.3),
            ], p=0.4),
            albumentations.HueSaturationValue(p=0.3),
        ], p=p)

    augmentation = _strong_aug(p=0.9)
    files = glob.glob(args.data_set + '/masks/*.*')
    for i in range(len(files)):
        mask = files[i]
        img = os.path.basename(mask)
        img = args.data_set + '/images/' + img
        files[i] = (img, mask)
    coco_images = _coco_bg(args)

    def _generator():
        for i in files:
            img = cv2.imread(i[0])
            mask = cv2.imread(i[1])
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img,mask = random_choice(img,mask,args.resolution)
            data = {"image": img, "mask": mask}
            augmented = augmentation(**data)
            img, mask = augmented["image"], augmented["mask"]

            fg = cv2.resize(img,(args.resolution, args.resolution))
            mask = cv2.resize(mask,(args.resolution, args.resolution))
            name = random.choice(coco_images)
            bg = cv2.imread(name)
            bg = _crop_back(bg,args.resolution, args.resolution)
            fmask = mask.astype(np.float32) / 255
            fmask = np.expand_dims(fmask, 2)
            fg = fg.astype(np.float32) / 255 * fmask
            bg = bg.astype(np.float32) / 255 * (1 - fmask)
            img = fg + bg
            img = np.clip(img, 0, 1)
            fg = np.clip(fg, 0, 1)
            bg = np.clip(bg, 0, 1)
            trimap = generate_trimap(mask)
            yield np.concatenate([img, trimap], axis=2), np.concatenate([fmask, fg, bg], axis=2)

    ds = tf.data.Dataset.from_generator(_generator, (tf.float32, tf.float32),
                                        (tf.TensorShape([args.resolution, args.resolution, 4]),
                                         tf.TensorShape([args.resolution, args.resolution, 7])))
    if training:
        ds = ds.shuffle(args.batch_size * 3, reshuffle_each_iteration=True)

    ds = ds.batch(args.batch_size, True)

    return ds
