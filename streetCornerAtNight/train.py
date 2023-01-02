# label:color_rgb:parts:actions
# background:0,0,0::
# space:140,120,240:: 113
# trapan:250,50,83:: 139
# -------------------------------Imports--------------------------------#
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ssl
import Epoch
import torch
from DatasetLoad import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from PreprocessAndAugmentation import augmentation, preprocess

ssl._create_default_https_context = ssl._create_unverified_context


def if_not_exist_create(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -------------------------------Data Locations--------------------------------#
directory = "dataset"
x_train_dir = directory + "/Train" + "/X"
y_train_dir = directory + "/Train" + "/y"
x_valid_dir = directory + "/Test" + "/X"
y_valid_dir = directory + "/Test" + "/y"
roi_mask_dir = directory + "/ROI.bmp"

# -------------------------------Model Settings--------------------------------#

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Motion', 'Unknown motion']
CLASS_VALUES = [255, 170]
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
INPUT_SHAPE = (576, 224) # Original (595, 245) Expected image height and width divisible by 32
AUGMENTATION_COUNT = 10

# -------------------------------Loss Settings--------------------------------#
loss = smp.utils.losses.DiceLoss()
LEARNING_RATE = 0.00001

# -------------------------------Model--------------------------------#

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# -------------------------------Train Dataset--------------------------------#
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    img_size=INPUT_SHAPE,
    classes=CLASSES,
    class_values=CLASS_VALUES,
    aug_count=AUGMENTATION_COUNT,
    aug_fn=augmentation,
    preprocess_fn=preprocess,
    roi_mask_dir=roi_mask_dir
)

# -------------------------------Validation Dataset--------------------------------#
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    img_size=INPUT_SHAPE,
    classes=CLASSES,
    class_values=CLASS_VALUES,
    preprocess_fn=preprocess,
    roi_mask_dir=roi_mask_dir
)

# -------------------------------Data loaders--------------------------------#
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# -------------------------------Epoch Settings--------------------------------#
EPOCHS = 50
max_score = 0
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore()
]

optimizer = torch.optim.RMSprop([
    dict(params=model.parameters(), lr=LEARNING_RATE),
])

train_epoch = Epoch.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = Epoch.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

model_name = input("Enter a model name: ")
if_not_exist_create("Models/")
if_not_exist_create("Models/" + model_name)

file_txt = open("Models/" + model_name + "/" + model_name + ".txt", "a")
log = ['Epochs: {}\n'.format(EPOCHS), 'Learning Rate: {}\n'.format(LEARNING_RATE), 'Encoder: {}\n'.format(ENCODER),
       'Encoder Weights: {}\n'.format(ENCODER_WEIGHTS), 'Activation: {}\n'.format(ACTIVATION)]
file_txt.writelines(log)
file_txt.close()

# -------------------------------Training--------------------------------#
for i in range(EPOCHS):
    file_txt = open("Models/" + model_name + "/" + model_name + ".txt", "a")
    log = []

    print('\nEpoch: {}'.format(i + 1))
    log.append('\nEpoch: {}\n'.format(i + 1))

    training_epoch_path = "Results/" + model_name + "/Train/epoch_" + str(i + 1)
    result_epoch_path = "Results/" + model_name + "/Valid/epoch_" + str(i + 1)

    if_not_exist_create(training_epoch_path)
    if_not_exist_create(result_epoch_path)

    train_logs = train_epoch.run(train_loader, training_epoch_path)
    valid_logs = valid_epoch.run(valid_loader, result_epoch_path)

    log.append("Training:\n")
    log.append(f"iou_score: {str(train_logs['iou_score'])} \n")
    log.append(f"dice_loss: {str(train_logs['dice_loss'])}\n")
    log.append(f"fscore: {str(train_logs['fscore'])}\n")
    log.append("Validation:\n")
    log.append(f"iou_score: {str(valid_logs['iou_score'])}\n")
    log.append(f"dice_loss: {str(valid_logs['dice_loss'])}\n")
    log.append(f"fscore: {str(valid_logs['fscore'])}\n")

    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model.state_dict(), 'Models/' + model_name + '/best_epoch.pth')
        print("Best Epoch Accuracy!\n")
        log.append("Best Epoch Accuracy!\n")

    # if i % 10 == 9:
    #     LEARNING_RATE /= 10
    #     optimizer.param_groups[0]['lr'] = LEARNING_RATE
    #     log.append(f" --> Learning rate decreased to {LEARNING_RATE} <--\n")

    file_txt.writelines(log)
    file_txt.close()

    print("*" * 110)
