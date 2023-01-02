import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from glob import glob


# Don't change this function
def default_augmentation(x, y, aug_count, preprocess_fn, img_size):
    x_a = preprocess_fn(x, img_size, 1)
    y_a = preprocess_fn(y, img_size, 0)

    return np.expand_dims(np.array(x_a), 0), np.expand_dims(np.array(y_a), 0)


def default_preprocess(x, img_size, type):
    return x


class Dataset(BaseDataset):


    def __init__(
            self,
            images_dir,
            masks_dir,
            img_size,
            classes=None,
            class_values=None,
            aug_count=1,
            aug_fn=default_augmentation,
            preprocess_fn=default_preprocess,
            roi_mask_dir=None
    ):
        self.images_fps = glob(images_dir + "/*")
        self.makes_fps = glob(masks_dir + "/*")

        self.classes = classes
        self.class_values = class_values
        self.aug_count = aug_count
        self.aug_fn = aug_fn
        self.preprocess_fn = preprocess_fn
        self.img_size = img_size
        self.roi_mask_dir = roi_mask_dir

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        masks = cv2.imread(self.makes_fps[i], 0)

        if self.roi_mask_dir is not None:
            roi_mask = cv2.imread(self.roi_mask_dir)
            image = np.where(roi_mask == 255, image, roi_mask)

        masks = cv2.resize(masks, (self.img_size[0], self.img_size[1]))

        mask = np.zeros((self.img_size[1], self.img_size[0], len(self.classes)))

        for row in range(mask.shape[0]):
            for column in range(mask.shape[1]):
                if masks[row][column] == self.class_values[0]:
                    mask[row][column][0] = 1
                elif masks[row][column] == self.class_values[1]:
                    mask[row][column][1] = 1

        aug_x, aug_y = self.aug_fn(image, mask, self.aug_count, self.preprocess_fn, self.img_size)

        return aug_x, aug_y

    def __len__(self):
        return len(self.images_fps)
