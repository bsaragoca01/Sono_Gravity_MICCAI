import argparse
import logging
import os
import random
import sys
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm
import torchvision.models.segmentation as segmentation
import numpy as np
from openpyxl import Workbook
from PIL import Image

from evaluate_w_metrics import evaluate_end_of_epoch
from data_loading import BasicDataset, CarvanaDataset, CustomCompose, RandomRotation, RandomHorizontalFlip, ColorJitter, AddGaussianNoiseNumpy
from dice_score import dice_loss, accuracy, precision, sensitivity, specificity, false_positive_rate, false_negative_rate

dir_img_path = os.path.expanduser('path_to_images_to_train')
dir_mask_path = os.path.expanduser('path_to_masks_to_train')
dir_checkpoint_path = os.path.expanduser('path_to_checkpoints')
dir_imgT_path = os.path.expanduser('path_to_validation_images')
dir_maskT_path = os.path.expanduser('path_to_validation_masks')

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
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

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
    #The user must uncomment the optimizer wanted to be use
    #optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer: SGD
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer: RMSprop
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Reduces the learning rate with a tolerance of 5 epochs when an improvement fo accuracy is lacking.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) 
    criterion = nn.CrossEntropyLoss() if model.classifier[4].out_channels > 1 else nn.BCEWithLogitsLoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.classifier[4].out_channels > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    #Excel workbook and worksheet setup for metrics
    wb = Workbook()
    ws = wb.active
    ws.append([
        'Epoch', 'Learning rate', 'Train Loss', 'Validation loss', 'Validation Dice Score', 'Validation Accuracy', 'Validation Precision',
        'Validation Sensitivity', 'Validation Specificity', 'Validation FPR', 'Validation FNR'
    ])

    #Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        #Initialize tqdm for training progress bar
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                assert images.dim() == 4, f"Expected 4-dimensional input, got {images.shape}"
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    output = model(images)['out']
                    masks_pred = output.argmax(dim=1)
                    if model.classifier[4].out_channels == 1:
                        loss = criterion(output.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(output.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(output, true_masks)
                        loss += dice_loss(
                            F.softmax(output, dim=1).float(),
                            F.one_hot(true_masks, model.classifier[4].out_channels).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        for class_idx in range(1, model.classifier[4].out_channels):
                            pred_mask_class = (output.argmax(dim=1) == class_idx).float()
                            true_mask_class = (true_masks == class_idx).float()

                optimizer.zero_grad(set_to_none=True) 
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})           
            save_images_side_by_side(images, true_masks, masks_pred, epoch)
            pbar.close()
            #Validation at the end of epoch
            val_metrics = evaluate_end_of_epoch(model, val_loader, device, amp, args.n_classes, criterion)

            current_lr = optimizer.param_groups[0]['lr']  #Get current learning rate
            #Log metrics
            logging.info(f'Epoch {epoch}/{epochs}, '
                         f'Train Loss: {epoch_loss / len(train_loader):.4f}, '
                         f'Validation loss: {val_metrics["avg_loss"]:.4f}, '
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

        #Saves the checkpoints every 5 epoch. it can be modified
        if epoch%5==0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
            checkpoint_path = dir_checkpoint / checkpoint_filename
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values 
            torch.save(state_dict, str(checkpoint_path))
            logging.info(f'Checkpoint saved: {checkpoint_path}')


    save_dir = os.path.expanduser('path_to_save_the_worksheet')
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    wb.save(Path(save_dir) / 'training_metrics_final_augmentation.xlsx')


def save_images_side_by_side(images, true_masks, masks_pred, epoch, save_dir='path_to_save_the_images'):
    images = [TF.to_pil_image(image.cpu().detach().squeeze()) for image in images]
    true_masks = [apply_color_map(mask.cpu().detach().numpy()) for mask in true_masks]
    masks_pred = [mask_to_pil(mask.argmax(dim=1)) for mask in masks_pred]
    save_dir = os.path.expanduser(save_dir) 
    os.makedirs(save_dir, exist_ok=True)
    # Save images side by side
    batch_size = len(images)
    for idx in range(batch_size):
        combined_image = Image.new('RGB', (images[idx].width * 3, images[idx].height))
        combined_image.paste(images[idx], (0, 0))
        combined_image.paste(true_masks[idx], (images[idx].width, 0))
        combined_image.paste(masks_pred[idx], (images[idx].width * 2, 0))
        combined_image.save(os.path.join(save_dir, f'epoch_{epoch}_image_{idx}.png'))
    print(f"Images saved to {save_dir}")

def mask_to_pil(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().squeeze().numpy()
    if mask.ndim == 2:  #Grayscale mask
        mask = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask, mode='L')
    elif mask.ndim == 3:  #Multi-class mask
        mask_argmax = np.argmax(mask, axis=0)  #Take the channel with highest value as predicted class
        mask_max = mask_argmax.max()
        if mask_max != 0:
            mask = (mask_argmax * 255 / mask_max).astype(np.uint8)
        else:
            mask = (mask_argmax * 255).astype(np.uint8)  #Handle the case where mask_max is 0
        return Image.fromarray(mask, mode='L')
    else:
        raise ValueError("Unsupported mask dimensions. Supported dimensions: 2 (single channel) or 3 (multi-channel)")

def apply_color_map(mask):
    cmap = cm.get_cmap('jet')  
    normed_data = mask / mask.max()  #Normalize mask to [0, 1]
    mapped_data = cmap(normed_data)  #Apply color map
    mapped_data = (mapped_data[:, :, :3] * 255).astype(np.uint8)  #Convert to RGB
    return Image.fromarray(mapped_data)

#Mandatory to use the --load in the command line by indicating the directory to the model intended to be fine-tuned.
def get_args():
    parser = argparse.ArgumentParser(description='Train the Deeplab model on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--n_classes', '-c', type=int, default=3, help='Number of classes in the dataset')

    return parser.parse_args()

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
            learning_rate=args.learning_rate,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.CudaError:
        torch.cuda.empty_cache()
        raise
