import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d, EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.networks.nets import UNETR,SwinUNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
import numpy as np
import SimpleITK as sitk

import torch
image_size = 96
num_samples = 4

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_config()
root_dir = "/data/MING/data/FLARE"
print(root_dir)
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
    ]
)

data_path = "your falre.json path "

val_files = load_decathlon_datalist(data_path, True, "FLARE_val")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0,num_workers=2
)

val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, pin_memory=True,num_workers=2
)



model = SwinUNETR(
    img_size=(image_size, image_size, image_size),
    in_channels=1,
    out_channels=15,
    feature_size=48,
    use_checkpoint=True,
).cuda()

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
cls_loss_function = torch.nn.CrossEntropyLoss()

torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
def calculate_surface_dice(segmentation_array, reference_array):

    # Compute boundary masks
    segmentation_boundary = compute_boundary_mask(segmentation_array)
    reference_boundary = compute_boundary_mask(reference_array)

    # Compute volumes
    segmentation_volume = np.sum(segmentation_array)
    reference_volume = np.sum(reference_array)

    # Compute NSD
    nsd = 1.0 - (2.0 * np.sqrt(np.sum(segmentation_boundary * reference_boundary))) / (np.sum(segmentation_boundary) + np.sum(reference_boundary)) * np.abs(segmentation_volume - reference_volume) / (segmentation_volume + reference_volume)

    return nsd

def compute_boundary_mask(binary_image):
    # Compute boundary mask using 26-connectivity
    boundary = np.zeros_like(binary_image)
    kernel = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]], dtype=np.uint8)
    for z in range(binary_image.shape[0]):
        boundary[z, :, :] = np.logical_xor(binary_image[z, :, :], np.logical_and(binary_image[z, :, :], np.logical_not(np.array(binary_image[z, :, :].astype(bool), dtype=np.uint8))))
    return boundary



def avg_dsc(source_mask, target_mask, binary=False, topindex=2, botindex=0,
            pad=[0, 0, 0], return_mean=True, detach=False):

    if not detach:
        target_mask = target_mask
    else:
        target_mask = target_mask.detach()

    standard_loss_sum = 0

    if binary:
        label = (torch.argmax(source_mask, dim=1, keepdim=True)).type(torch.LongTensor)
        one_hot = torch.FloatTensor(label.size(0), source_mask.size(1), label.size(2), label.size(3),
                                         label.size(4)).zero_()
        source_mask = one_hot.scatter_(1, label.data, 1)
        label = (torch.argmax(target_mask, dim=1, keepdim=True)).type(torch.LongTensor)
        one_hot = torch.FloatTensor(label.size(0), target_mask.size(1), label.size(2), label.size(3),
                                         label.size(4)).zero_()
        target_mask = one_hot.scatter_(1, label.data, 1)
    else:
        source_mask = source_mask

    if source_mask.shape[1] > 1:
        # standard_loss_sum = standard_loss_sum + dice(source_mask[:,1:2,...],target_mask[:,1:2,...])
        # standard_loss_sum = standard_loss_sum + dice(source_mask[:,2:3,...],target_mask[:,2:3,...])
        if return_mean:
            standard_loss_sum += torch.mean((2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001))[:,
                                            botindex:topindex, ...])
        else:
            standard_loss_sum += torch.mean((2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001))[:,
                                            botindex:topindex, ...], 1)
    else:
        if return_mean:
            standard_loss_sum += torch.mean(2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001))
        else:
            standard_loss_sum += torch.mean(2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001), 1)
    return standard_loss_sum


def validation(epoch_iterator_val):
    model.eval()
    nsdlist = []
    dicelist = []
    liver = []
    rk = []
    spleen = []
    pan = []
    aorta = []
    IVC = []
    RAG = []
    LAG = []
    Gallbladder = []
    eso = []
    stomatch = []
    duodenum = []
    lk = []
    tumor = []
    dice_total_list = [[], [], [], [],[], [],[], [],[], [],[], [],[], [],]
    nsd_total_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], ]
    with torch.no_grad():
        for idx, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dicelist.append(dice_metric)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))

            del val_outputs_list
            del val_labels
            del val_inputs
            del val_outputs
            del val_labels_list
            for j in range(1, 15):
                dice = avg_dsc(val_output_convert[0].unsqueeze(0).detach().cpu(), val_labels_convert[0].unsqueeze(0).detach().cpu(), botindex=j, topindex=j+1).item()
                for i in range(val_labels_convert[0].shape[-1]):
                    xx = val_labels_convert[0][j, ..., i].unsqueeze(dim=0).unsqueeze(dim=0).detach().cpu()
                    yy = val_output_convert[0][j, ..., i].unsqueeze(dim=0).unsqueeze(dim=0).detach().cpu()
                    nsd(xx, yy)
                nsd_val = nsd.aggregate()[0].item()
                dice_total_list[j-1].append(dice)
                nsd_total_list[j-1].append(nsd_val)

        mean_dice_val = dice_metric.aggregate().item()
        nsd_mean = 0
        dice_metric.reset()
    return dice_total_list, mean_dice_val, nsd_total_list, nsd_mean

max_iterations = 80000
eval_num = 250
post_label = AsDiscrete(to_onehot=15)
post_pred = AsDiscrete(argmax=True, to_onehot=15)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
nsd = SurfaceDiceMetric(class_thresholds=[2.5],  get_not_nans=True)

global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
nsd_values = []
# load your model path
# model.load_state_dict(torch.load(os.path.join(root_dir, "swinunter_best_metric_model8.pth")))
model_path = "your model path"
model.load_state_dict(torch.load(model_path))
print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")
epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
dice_list, dice_val, nsdlist, nsd_mean = validation(epoch_iterator_val)
metric_values.append(dice_val)



