import json
import random
import cv2
import numpy as np

class CocoBG:
    def __init__(self,path):
        self.images = _load_coco(path)


    def get_random(self,w,h,rgb=True):
        name = random.choice(self.images)
        bg = cv2.imread(name)
        if rgb:
            bg = bg[:,:,::-1]
        return self.crop_back(bg, w, h)

    def crop_back(self,img, w, h):
        nw = max(img.shape[1], w)
        nh = max(img.shape[0], h)
        img = cv2.resize(img, (nw, nh))
        x_shift = int(np.random.uniform(0, nw - w))
        y_shift = int(np.random.uniform(0, nh - h))
        return img[y_shift:y_shift + h, x_shift:x_shift + w, :]


def _load_coco(coco_path):
    if coco_path == 'test':
        return ['./testdata/default.png']
    with open(coco_path + '/annotations/instances_train2017.json') as f:
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
        name = '{}/train2017/{:012d}.jpg'.format(coco_path, int(k))
        names.append(name)
    return names