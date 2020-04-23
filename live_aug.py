import cv2
import numpy as np

original_img = cv2.imread('testdata/images/test.png')
original_mask = cv2.imread('testdata/masks/test.png')
bg = cv2.imread('testdata/default.png')
original_img = cv2.resize(original_img,(160,160))
original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2GRAY)
original_mask = cv2.resize(original_mask,(160,160))
bg = cv2.resize(bg,(160,160))
bg = bg.astype(np.float32)

def _strong_aug(p=0.5):
    import albumentations
    return albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=0, p=0.5,border_mode=cv2.BORDER_CONSTANT),
        albumentations.OneOf([
            albumentations.OpticalDistortion(p=0.5,border_mode=cv2.BORDER_CONSTANT),
            albumentations.GridDistortion(p=0.5,border_mode=cv2.BORDER_CONSTANT),
            albumentations.IAAPiecewiseAffine(p=0.5),
            albumentations.ElasticTransform(p=0.5,border_mode=cv2.BORDER_CONSTANT),
        ], p=0.5),
        albumentations.OneOf([
            albumentations.CLAHE(clip_limit=2),
            albumentations.IAASharpen(),
            albumentations.IAAEmboss(),
        ], p=0.5),
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(p=0.5),
        ], p=0.4),
        albumentations.HueSaturationValue(p=0.5),
    ], p=p)

augmentation = _strong_aug(p=1)

while True:
    data = {"image": original_img, "mask": original_mask}
    augmented = augmentation(**data)
    img, mask = augmented["image"], augmented["mask"]
    img = img.astype(np.float32)
    mask = mask.astype(np.float32) / 255
    mask = np.expand_dims(mask, axis=2)
    img = img * mask + bg * (1 - mask)
    img = img.astype(np.uint8)
    cv2.imshow('test',img)
    key = cv2.waitKey()
    if key in [ord('q'), 202]:
        break