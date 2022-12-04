import torch
import random
import numpy as np
import numbers
from PIL import Image  # PyTorch likes PIL instead of cv2
import cv2

# the code is from Chris Xie (https://github.com/chrisdxie/uois)

##### Useful Utilities #####

def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4:  # NHWC
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    elif array.ndim == 3:  # HWC
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    else:  # everything else
        tensor = torch.from_numpy(array).float()

    return tensor


def translate(img, tx, ty, interpolation=cv2.INTER_LINEAR):
    """ Translate img by tx, ty

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    M = np.array([[1, 0, tx],
                  [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)


def rotate(img, angle, center=None, interpolation=cv2.INTER_LINEAR):
    """ Rotate img <angle> degrees counter clockwise w.r.t. center of image

        @param img: a [H x W x C] image (could be an RGB image, flow image, or label image)
    """
    H, W = img.shape[:2]
    if center is None:
        center = (W // 2, H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, M, (W, H), flags=interpolation)


##### Depth Augmentations #####

def add_noise_to_depth_cuda(depth_img, noise_params):
    """ Distort depth image with multiplicative gamma noise.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    # Multiplicative noise: Gamma random variable
    multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
    depth_img = multiplicative_noise * depth_img
    return depth_img

def dropout_random_ellipses_mask(mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(mask > 0)).T  # Shape: [#nonzero_pixels x 2]

    if nonzero_pixel_indices.shape[0] == 0 or num_ellipses_to_dropout == 0:
        return mask

    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]  # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'],
                              size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'],
                              size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :].astype(int)
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        mask_drop = np.zeros_like(mask).astype(np.float32)
        mask_drop = cv2.ellipse(mask_drop, tuple(center[::-1]), (x_radius, y_radius), angle, 0, 360,
                           color=1, thickness=-1)
        mask[mask_drop == 1] = 0

    return mask