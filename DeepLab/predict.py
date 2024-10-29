import argparse
import logging
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models.segmentation as segmentation
from openpyxl import Workbook
from data_loading import CarvanaDataset
from evaluate_w_metrics import evaluate_per_image
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import time

#Directory paths
dir_img_path = os.path.expanduser('path_to_the_test_data_images')
dir_mask_path = os.path.expanduser('path_to_the_test_data_masks')
dir_checkpoint_path = os.path.expanduser('path_to_save_checkpoints')
dir_img = Path(dir_img_path)
dir_mask = Path(dir_mask_path)
dir_checkpoint = Path(dir_checkpoint_path)

def test_model(
        model,
        device,
        batch_size: int = 1,
        amp: bool = False,
        img_scale: float = 0.5,
):
    #Create dataset
    datasetTest = CarvanaDataset(dir_img, dir_mask, img_scale)

    #Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False, drop_last=True)
    val_loader = DataLoader(datasetTest, shuffle=False, **loader_args)
    wb = Workbook()
    ws = wb.active
    headers = [
        'Image', 'Validation Dice Score', 'Validation Accuracy', 'Validation Precision', 'Validation Sensitivity', 'Validation Specificity', 'Validation FPR', 'Validation FNR', 'Prediction time',
    ]
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_loader) * batch_size) as progress_bar:
        for i, batch in enumerate(val_loader):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            start_time = time.time()
            assert images.dim() == 4, f"Expected 4-dimensional input, got {images.shape}"
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                output = model(images)['out']
                masks_pred = output.argmax(dim=1)
            end_time = time.time()
            prediction_time = end_time - start_time
            save_images_side_by_side(images, true_masks, masks_pred, i)
            metrics = evaluate_per_image(model, images[0].unsqueeze(0), true_masks[0].unsqueeze(0), device, amp, args.classes)
            row = [
                i,
                metrics["avg_dice_score"],
                metrics["avg_accuracy"],
                metrics["avg_precision"],
                metrics["avg_sensitivity"],
                metrics["avg_specificity"],
                metrics["avg_fpr"],
                metrics["avg_fnr"],
                prediction_time
            ]
            progress_bar.update(1)
    save_dir = os.path.expanduser('path_to_save_the_worksheet')
    os.makedirs(save_dir, exist_ok=True)
    wb.save(os.path.join(save_dir, 'training_metrics_final.xlsx'))


def save_images_side_by_side(images, true_masks, masks_pred, img, save_dir='path_to_save_the_images'):
    images = [TF.to_pil_image(image.cpu().detach().squeeze()) for image in images]
    true_masks_pil = [mask_color(true_mask.cpu().detach().numpy()) for true_mask in true_masks]
    masks_pred = [mask_color(mask.cpu().detach().numpy()) for idx, mask in enumerate(masks_pred)]
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(len(images)):
        combined_image = Image.new('RGB', (images[idx].width * 3, images[idx].height))
        combined_image.paste(images[idx], (0, 0)) #original image
        combined_image.paste(true_masks_pil[idx], (images[idx].width, 0)) #ground-truth mask
        combined_image.paste(masks_pred[idx], (images[idx].width * 2, 0)) #predicted mask
        combined_image.save(os.path.join(save_dir, f'image_{img}_{idx}.png'))
    print(f"Images saved to {save_dir}")


def mask_color(pred_mask):
    color_map = {
        0: (0,0,0),
        1: (255,255,255),
        2: (128,128,128)
    }
    height, width = pred_mask.shape[-2], pred_mask.shape[-1]
    color_mapped_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_value, color in color_map.items():
        color_mapped_image[pred_mask == class_value] = color
    return Image.fromarray(color_mapped_image)


def get_args():
    parser = argparse.ArgumentParser(description='Test the Deeplab model on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    return parser.parse_args()
#The model MUST be loaded to segment the images. Thus, --load must be used in the command

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = segmentation.deeplabv3_resnet50(pretrained=False) 
    #If it is intended to use DeepLab with ResNet-101 backbone, use the command:
    #model=segmentation.deeplabv3_resnet101(pretrained=False) 
    model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #Modification in the first layer of model's architecture to allow the input of 1-color channel data
    model.classifier[4] = nn.Conv2d(256, args.n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.to(memory_format=torch.channels_last)
    model.to(device=device)

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        state_dict = {k: v for k, v in checkpoint.items() if k in model.state_dict()}
        model.load_state_dict(state_dict, strict=False)
        logging.info(f'Model loaded from {args.load}')

    try:
        test_model(model=model, device=device, batch_size=args.batch_size, amp=args.amp, img_scale=args.scale)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
