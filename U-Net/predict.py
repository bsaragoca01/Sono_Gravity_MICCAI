import argparse
import logging
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from openpyxl import Workbook
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from evaluate_metrics import evaluate_per_image
from unet import UNet
from data_loading import CarvanaDataset

#Directory paths
#Images, intended to be predicted by U-Net, and respective ground-truth masks directories must be introduced

dir_img_path = os.path.expanduser("images_path")
dir_mask_path = os.path.expanduser("masks_path")
dir_img = Path(dir_img_path)
dir_mask = Path(dir_mask_path)


def mask_color(pred_mask):
    #Define the color map for the classes
    color_map = {
        0: (0, 0, 0),      #Class 0 (background)
        1: (255, 255, 255), #Class 1 (Bone)
        2: (128, 128, 128)  #Class 2 (Ligament)
    }

    height, width = pred_mask.shape[-2], pred_mask.shape[-1]
    color_mapped_image = np.zeros((height, width, 3), dtype=np.uint8)

    #Map each class value to its corresponding color
    for class_value, color in color_map.items():
        color_mapped_image[pred_mask == class_value] = color

    return Image.fromarray(color_mapped_image)

def save_images_side_by_side(images, true_masks, masks_pred, img_name, save_dir='path_to_save_images'):
    images = [TF.to_pil_image(image.cpu().detach().squeeze()) for image in images]
    true_masks_np = [mask_color(mask.cpu().detach().numpy()) for mask in true_masks]
    masks_pred_np = [mask_color(pred_mask.argmax(dim=0).cpu().detach().numpy()) for pred_mask in masks_pred]

    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(len(images)):
        img_width, img_height = images[idx].size
        true_mask_width, true_mask_height = true_masks_np[idx].size
        pred_mask_width, pred_mask_height = masks_pred_np[idx].size

        #The dimensions between the images, ground-truth masks and predictions are checked
        assert (img_width, img_height) == (true_mask_width, true_mask_height), \
            f"Dimension mismatch: Image ({img_width}, {img_height}) vs True Mask ({true_mask_width}, {true_mask_height})"
        assert (img_width, img_height) == (pred_mask_width, pred_mask_height), \
            f"Dimension mismatch: Image ({img_width}, {img_height}) vs Pred Mask ({pred_mask_width}, {pred_mask_height})"

        
        combined_image = Image.new('RGB', (img_width * 3, img_height))
        combined_image.paste(images[idx], (0, 0)) #original image
        combined_image.paste(true_masks_np[idx], (img_width, 0)) #ground-truth mask
        combined_image.paste(masks_pred_np[idx], (img_width * 2, 0)) #predicted mask

        #Use a consistent file name
        file_name = f'image_{img_name}_{idx}.png'
        combined_image.save(os.path.join(save_dir, file_name))

    print(f"Images saved to {save_dir}")


def test_model(model, device, batch_size: int = 1, amp: bool = False, img_scale: float = 0.5):

    datasetTest = CarvanaDataset(dir_img, dir_mask, img_scale)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False)
    val_loader = DataLoader(datasetTest, shuffle=False, **loader_args)

    wb = Workbook()
    ws = wb.active
    headers = [
        'Image', 'Validation Dice Score', 'Validation Accuracy', 'Validation Precision', 'Validation Sensitivity', 'Validation Specificity', 'Validation FPR', 'Validation FNR', 'Prediction Time'
    ]
    ws.append(headers)
    model.eval() #used to validate

    with torch.no_grad(), tqdm(total=len(val_loader) * batch_size) as progress_bar:
        for i, batch in enumerate(val_loader):
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            start_time = time.time()  #Start timer to measure the prediction time
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                if model.n_classes > 1:
                    pred_mask = masks_pred.argmax(dim=1).cpu().detach().numpy()
                else:
                    pred_mask = (F.sigmoid(masks_pred) > 0.5).float().cpu().detach().numpy() 
                end_time = time.time()  #End timer, the prediction is over

                prediction_time = end_time - start_time  #Calculate prediction time
                save_images_side_by_side(images, true_masks, masks_pred, i) #if the user does not want to save the images, then must comment this line
                metrics = evaluate_per_image(model, images[0].unsqueeze(0), true_masks[0].unsqueeze(0), device, amp)
                row = [
                    i,
                    metrics["dice_score"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["sensitivity"],
                    metrics["specificity"],
                    metrics["fpr"],
                    metrics["fnr"],
                    prediction_time,
                ]
                ws.append(row)
                progress_bar.update(1)

    save_dir = os.path.expanduser('path_to_save_the_worksheet_with_the_metrics')
    os.makedirs(save_dir, exist_ok=True)
    wb.save(os.path.join(save_dir, 'training_metrics_final.xlsx'))

def get_args():
    parser = argparse.ArgumentParser(description='Test the U-Net on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    return parser.parse_args()
#The model MUST be loaded to segment the images. Thus, --load must be used in the command

def main():
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')

    #Load model weights if provided
    if args.load:
        if os.path.isfile(args.load):
            state_dict = torch.load(args.load, map_location=device)
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load}')
        else:
            logging.error(f'File {args.load} not found. Exiting.')
            return

    model.to(device=device)

    try:
        test_model(
            model=model,
            device=device,
            batch_size=args.batch_size,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f'An unexpected error occurred: {e}')
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

