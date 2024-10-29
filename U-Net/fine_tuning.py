import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from openpyxl import Workbook
from PIL import Image
from evaluate_metrics import evaluate_end_of_epoch
from unet import UNet
from data_loading import BasicDataset, CarvanaDataset, CustomCompose, RandomRotation, RandomHorizontalFlip, ColorJitter, AddGaussianNoiseNumpy

dir_img_path = os.path.expanduser('Data/Train/images')
dir_mask_path = os.path.expanduser('Data/Train/masks')
dir_checkpoint_path = os.path.expanduser('checkpoint_path')
dir_imgT_path = os.path.expanduser('Data/Validation/images')
dir_maskT_path = os.path.expanduser('Data/Validation/images')

dir_img = Path(dir_img_path)
dir_mask = Path(dir_mask_path)
dir_checkpoint = Path(dir_checkpoint_path)
dir_imgT = Path(dir_imgT_path)
dir_maskT = Path(dir_maskT_path)

#Data Augmentation: it must be changed, according to the types of transformations intended to be use
custom_transform = CustomCompose([
    RandomRotation(degrees=20), #range of left random rotation
    RandomHorizontalFlip(), 
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #color adjustments
    AddGaussianNoiseNumpy(mean=0, std=0.1) #gaussian noise
])

def train_model(
        model,
        device,
        epochs: int = 100,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-8, #L2 regularization
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    #Create dataset
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale, transform = custom_transform)
    datasetTest = CarvanaDataset(dir_imgT, dir_maskT, img_scale, transform = None) # the transformations are not applied to the validation set

    n_val = len(datasetTest)
    n_train = len(dataset)
    train_set, val_set = dataset, datasetTest

    #Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    #Logg information about the training process

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')


    #Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #The user must uncomment the optimizer it wants to use
    
    #optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer: SGD
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer: RMSprop
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Reduces the learning rate with a tolerance of 5 epochs when an improvement fo accuracy is lacking.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    #Excel workbook and worksheet creation
    wb = Workbook()
    ws = wb.active
    ws.append([
        'Epoch', 'Learning rate','Train Loss', 'Validation Loss', 'Validation Dice Score', 'Validation Accuracy', 'Validation Precision',
        'Validation Sensitivity', 'Validation Specificity', 'Validation FPR', 'Validation FNR'
    ])

    #Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        #Initialize tqdm for training progress bar visualization
        pbar_train = tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img', leave=True)

        for batch_idx, batch in enumerate(train_loader):
            images, true_masks = batch['image'], batch['mask']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            masks_pred = model(images)

            loss = criterion(masks_pred, true_masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #Update tqdm progress bar
            pbar_train.set_postfix({'Train Loss': epoch_loss / (batch_idx + 1)})
            pbar_train.update(images.shape[0])  # Increment by batch size

        #Save images side by side at the end of each epoch
        #save_images_side_by_side(images, true_masks, masks_pred, epoch)
        pbar_train.close()

        #Validation at the end of epoch
        val_metrics = evaluate_end_of_epoch(model, val_loader, device, amp, criterion)

        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        #Log metrics

        logging.info(f'Epoch {epoch}/{epochs}, '
                     f'Train Loss: {epoch_loss / len(train_loader):.4f}, '
                     f'Validation Loss: {val_metrics["avg_loss"]:.4f}, '
                     f'Validation Dice Score: {val_metrics["avg_dice_score"]:.4f}, '
                     f'Validation Accuracy: {val_metrics["avg_accuracy"]:.4f}, '
                     f'Validation Precision: {val_metrics["avg_precision"]:.4f}, '
                     f'Validation Sensitivity: {val_metrics["avg_sensitivity"]:.4f}, '
                     f'Validation Specificity: {val_metrics["avg_specificity"]:.4f}, '
                     f'Validation FPR: {val_metrics["avg_fpr"]:.4f}, '
                     f'Validation FNR: {val_metrics["avg_fnr"]:.4f}')

        #Save metrics to Excel
        ws.append([
            epoch,
            current_lr,
            epoch_loss / len(train_loader),
            val_metrics["avg_loss"],
            val_metrics["avg_dice_score"],
            val_metrics["avg_accuracy"],
            val_metrics["avg_precision"],
            val_metrics["avg_sensitivity"],
            val_metrics["avg_specificity"],
            val_metrics["avg_fpr"],
            val_metrics["avg_fnr"]
        ])

        scheduler.step(val_metrics['avg_accuracy']) #changes the learning rate within 5 epochs if there is no improvement in the accuracy

        #Saves the checkpoint every 5 epoch. It can be modified.
        if epoch%5==0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
            checkpoint_path = dir_checkpoint / checkpoint_filename
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values  #if additional info needs to be saved
            torch.save(state_dict, str(checkpoint_path))
            logging.info(f'Checkpoint saved: {checkpoint_path}')    


    save_dir = os.path.expanduser('path_to_save_the_worksheet_with_the_metrics')
    #Ensure the directory exists; create it if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    wb.save(Path(save_dir) / 'training_metrics_final.xlsx')

def save_images_side_by_side(images, true_masks, masks_pred, epoch, save_dir='path_to_save_images'):
    #Convert images to PIL format
    images = [TF.to_pil_image(image.cpu().detach().squeeze()) for image in images]
    #Convert true masks to PIL format with custom color mapping
    true_masks = [apply_color_map(mask.cpu().detach().numpy()) for mask in true_masks]
    #Convert predicted masks to PIL format
    masks_pred = [apply_color_map(mask.argmax(dim=0).cpu().detach().numpy()) for mask in masks_pred]

    save_dir = os.path.expanduser(save_dir) 
    os.makedirs(save_dir, exist_ok=True)

    batch_size = len(images)
    for idx in range(batch_size):
        combined_image = Image.new('RGB', (images[idx].width * 3, images[idx].height))
        combined_image.paste(images[idx], (0, 0)) #original image
        combined_image.paste(true_masks[idx], (images[idx].width, 0)) #true mask
        combined_image.paste(masks_pred[idx], (images[idx].width * 2, 0)) #mask predicted

        combined_image.save(os.path.join(save_dir, f'epoch_{epoch}_image_{idx}.png'))

    print(f"Images saved to {save_dir}")

def apply_color_map(mask):
    color_map = {
        0: (0, 0, 0),       # Class 0: Black
        1: (255, 255, 255), # Class 1: White
        2: (128, 128, 128)  # Class 2: Grey
    }
    
    #Create an empty RGB image
    height, width = mask.shape[-2], mask.shape[-1]
    color_mapped_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    #Apply the color map
    for class_value, color in color_map.items():
        color_mapped_image[mask == class_value] = color
    
    return Image.fromarray(color_mapped_image)

#some parameters must be defined here on the default or in the command line

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

#--load command MUST be used in the comand line with the directory to the model's checkpoint intended to be fine-tuned.

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    #Loads the model intended to be fine-tuned
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model = model.to(device=device)

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Consider reducing batch size or image scale.')
        torch.cuda.empty_cache()
