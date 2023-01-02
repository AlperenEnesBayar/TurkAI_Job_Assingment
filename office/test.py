import os
from numpy import ndarray

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import Epoch
import torch
from DatasetLoad import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from glob import glob
from PreprocessAndAugmentation import augmentation, preprocess
from skimage.morphology import convex_hull_image


def one_channel_to_three(x):
    if x.shape[-1] != 3:
        x = cv2.merge((x, x, x))
    return x


def normalize_label(x):
    x = x.cpu().detach().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x1 = x[0]
        x2 = x[1]
        x1 = one_channel_to_three(x1) * 255.0
        x2 = one_channel_to_three(x2) * 255.0
        return x1, x2


def resize_img(img):
    return cv2.resize(img, (640, 512))


ENCODER = 'inceptionresnetv2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['front', 'sides']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
INPUT_SHAPE = (1216, 1024)
loss = smp.utils.losses.DiceLoss()

pipe_model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

pipe_model.load_state_dict(torch.load("inception.pth"))
kernel = np.ones((5, 5), np.uint8)
images = glob("D:\\Projects\\ArcelikBuzdolabi2022\\colored\\dataset\\Test\\X\\*")

for image_p in images:
    print(image_p)
    frame = cv2.imread(image_p)
    org_img = frame
    org_img = cv2.resize(org_img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img = org_img / 255.0
    x = np.moveaxis(img, -1, 0)
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)

    pipe_model = pipe_model.cuda()
    pred = pipe_model.predict(x.float().cuda())
    pred1, pred2 = normalize_label(pred)

    pred1 = cv2.resize(pred1, (frame.shape[1], frame.shape[0]))
    pred2 = cv2.resize(pred2, (frame.shape[1], frame.shape[0]))

    ret, pred1 = cv2.threshold(pred1, 120, 255, cv2.THRESH_BINARY)
    ret, pred2 = cv2.threshold(pred2, 120, 255, cv2.THRESH_BINARY)

    # pred1 = cv2.morphologyEx(pred1, cv2.MORPH_CLOSE, kernel)

    print("a")
    chull = convex_hull_image(pred1)
    print("b")

    cv2.imshow("pred1", pred1)
    cv2.imshow("chull", chull)

    frame = np.where(pred1 == 255, 255, frame)

    cv2.imshow("a.png", cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3)))
    cv2.waitKey(0)
