from scipy import ndimage

from fbamatting.transforms import trimap_transform, groupnorm_normalise_image
import torch
from fbamatting.models import build_model
import cv2
import numpy as np


def generate_trimap(alpha):
    trimap = np.copy(alpha)
    k_size = 15
    trimap[np.where((ndimage.grey_dilation(alpha[:, :], size=(k_size, k_size)) - ndimage.grey_erosion(alpha[:, :],
                                                                                                      size=(k_size,
                                                                                                            k_size))) != 0)] = 128
    return trimap


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def load_model(gpu, path):
    args = {'encoder': 'resnet50_GN_WS',
            'decoder': 'fba_decoder',
            'weights': path}
    return build_model(gpu, args)


def pred(model, image, mask, do_trimap=True):
    if do_trimap:
        pre_mask = generate_trimap(mask)
    else:
        pre_mask = mask
    trimap = np.zeros((pre_mask.shape[0], pre_mask.shape[1], 2))
    trimap[pre_mask == 255, 1] = 1
    trimap[pre_mask == 0, 0] = 1
    _, _, alpha = _pred(image, trimap, model)
    return alpha, pre_mask


def _pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    h, w = trimap_np.shape[:2]

    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

        output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

        output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]
    return fg, bg, alpha
