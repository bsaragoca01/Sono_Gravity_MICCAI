import argparse
import logging
import os
import torch
import time
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models.segmentation as segmentation
from openpyxl import Workbook
import numpy as np
from PIL import Image
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
from evaluate_w_margin import evaluate_per_image, dilate_mask, erode_mask
from data_loading import CarvanaDataset

#Directories must be replaced
dir_img_path = os.path.expanduser('path_to_the_test_data_images')
dir_mask_path = os.path.expanduser('path_to_the_test_data_masks')
dir_checkpoint_path = os.path.expanduser('path_to_save_checkpoints')
dir_img = Path(dir_img_path)
dir_mask = Path(dir_mask_path)
dir_checkpoint = Path(dir_checkpoint_path)

def test_model(model, device, batch_size: int = 1, amp: bool = False, img_scale: float = 0.5):
    datasetTest = CarvanaDataset(dir_img, dir_mask, img_scale)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False)
    val_loader = DataLoader(datasetTest, shuffle=False, **loader_args)
    wb = Workbook()
    ws = wb.active
    headers = [
        'Image','Count erodid','Count dilated', 'Count initial', 'Validation Dice Score', 'Validation Accuracy', 'Validation Precision', 'Validation Sensitivity',
        'Validation Specificity', 'Validation FPR', 'Validation FNR', 'Time'
    ]
    headers.extend([f'Dice_Class_{i}' for i in range(args.classes)])
    ws.append(headers)
    model.eval()

    with torch.no_grad(), tqdm(total=len(val_loader) * batch_size) as progress_bar:
        for i, batch in enumerate(val_loader):
            images, mask_true = batch['image'].to(device), batch['mask'].to(device)
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = mask_true.to(device=device, dtype=torch.long)
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                start_time = time.time() #monitoring of time to perform a prediction
                output = model(images)['out']
                finish_time = time.time()
                masks_pred = output.argmax(dim=1)
                dilated_masks = [dilate_mask(mask) for mask in true_masks]
                eroded_masks = [erode_mask(mask) for mask in true_masks]
                save_images_side_by_side(images, true_masks, dilated_masks, eroded_masks, masks_pred, i)
                metrics = evaluate_per_image(model, images[0].unsqueeze(0), true_masks[0].unsqueeze(0), args.classes, device, amp, apply_adjustment=True)
                row = [
                    i,
                    metrics['count_eroded_dice'], #to verify the impact of the margin appliance
                    metrics['count_dilated_dice'],
                    metrics['count_normal_dice'],
                    metrics['dice_score'],
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['sensitivity'],
                    metrics['specificity'],
                    metrics['fpr'],
                    metrics['fnr'],
                    finish_time - start_time
                ]

                for class_idx in range(0, args.classes):
                    row.append(metrics.get(f'class_{class_idx}_dice', 0))  #Default to 0 if the key is missing

                ws.append(row)

                progress_bar.update(batch_size)

    save_dir = os.path.expanduser('path_to_save_the_worksheet')
    os.makedirs(save_dir, exist_ok=True)
    wb.save(os.path.join(save_dir, 'training_metrics_final.xlsx'))

def save_images_side_by_side(images, true_masks, dilated_masks, eroded_masks, masks_pred, img_name, save_dir='path_to_save_predictions'):
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    #Convert tensors to PIL images only after processing
    images = [TF.to_pil_image(image.cpu().detach().squeeze()) for image in images]
    true_masks_np = [mask_color(mask.cpu().detach().numpy().astype(np.uint8)) for mask in true_masks]
    dilated_masks_np = [mask_color(mask.cpu().detach().numpy().astype(np.uint8)) for mask in dilated_masks]
    eroded_masks_np = [mask_color(mask.cpu().detach().numpy().astype(np.uint8)) for mask in eroded_masks]
    masks_pred_np = [mask_color(mask.cpu().detach().numpy()) for mask in masks_pred]
    for idx in range(len(images)):
        img_width, img_height = images[idx].size
        combined_image = Image.new('RGB', (img_width * 5, img_height))
        combined_image.paste(images[idx], (0, 0)) #original image
        combined_image.paste(true_masks_np[idx], (img_width, 0)) #ground-truth mask
        combined_image.paste(dilated_masks_np[idx], (img_width * 2, 0)) #ground-truth mask dilated
        combined_image.paste(eroded_masks_np[idx], (img_width * 3, 0)) #ground-truth mask eroded
        combined_image.paste(masks_pred_np[idx], (img_width * 4, 0)) #predicted mask

        file_name = f'image_{img_name}_{idx}.png'
        combined_image.save(os.path.join(save_dir, file_name))

    print(f"Images saved to {save_dir}")


def mask_color(pred_mask):
    color_map = {
        0: (0, 0, 0),      #Class 0 (background)
        1: (255, 255, 255), #Class 1 (Bone)
        2: (128, 128, 128)  #Class 2 (Ligament)
    }
    height, width = pred_mask.shape[-2], pred_mask.shape[-1]                                                                         color_mapped_image = np.zeros((height, width, 3), dtype=np.uint8)
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
    model.classifier[4] = nn.Conv2d(256, args.classes, kernel_size=(1, 1), stride=(1, 1))
    #Set the n_classes attribute
    model.n_classes = args.classes
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
