import cv2
import numpy as np
from PIL import Image
import albumentations as A


def preprocess(img, img_size, mode=1):
    img = cv2.resize(img, img_size)
    img = np.moveaxis(img, -1, 0)
    if mode == 1:
        img = img / 255.0
    return img


def augmentation(img, mask, aug_count, preprocess_fn, img_size):
    transform = A.Compose(
        [
            A.ShiftScaleRotate(rotate_limit=15, shift_limit=0.0625, scale_limit=0.1, p=0.25),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25)

        ],
        additional_targets={'image0': 'image'}
    )

    noise = A.Compose(
        [
            A.GaussNoise(var_limit=(2, 10), p=0.2),
            A.Blur(blur_limit=1, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2)
        ]
    )

    aug_x_list = []
    aug_y_list = []
    if aug_count == 1:
        aug_x = preprocess_fn(img, img_size, 1)
        aug_y = preprocess_fn(mask, img_size, 0)
        aug_x_list.append(aug_x)
        aug_y_list.append(aug_y)
    else:
        for _ in range(aug_count):
            aug_imgs = transform(image=img, image0=mask)
            second_aug_images = noise(image=aug_imgs["image"])

            aug_x = preprocess_fn(second_aug_images["image"], img_size, 1)
            aug_y = preprocess_fn(aug_imgs["image0"], img_size, 0)
            aug_x_list.append(aug_x)
            aug_y_list.append(aug_y)
    return np.array(aug_x_list), np.array(aug_y_list)
