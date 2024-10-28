from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import random
import torch

#Transformations used in data augmentation:

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img, angle), F.rotate(mask, angle)

class RandomHorizontalFlip:
    def __call__(self, img, mask):
        if random.random() > 0.5:
            return F.hflip(img), F.hflip(mask)
        return img, mask

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img, mask):
        return self.jitter(img), mask  

class AddGaussianNoiseNumpy:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img_array = np.array(img)  #Convert PIL image to numpy array
        noise = np.random.normal(self.mean, self.std, img_array.shape)  # Generate noise
        img_array = img_array / 255.0  #Normalize the image 
        img_array = img_array + noise  #Add noise to the image
        img_array = np.clip(img_array, 0, 1)  #Ensure that the values are within the range [0, 1]
        img_array = (img_array * 255).astype(np.uint8)  #Scale back to [0, 255]
        img = Image.fromarray(img_array)  #Convert back to PIL image
        return img, mask


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        tensor = torch.load(filename)
        if tensor.ndim == 2:  #Shape (H, W) for masks
            return Image.fromarray(tensor.numpy())
        elif tensor.ndim == 3 and tensor.size(0) in [1, 3]:  #Shape (C, H, W)
            return F.to_pil_image(tensor)
        else:
            raise ValueError(f'Loaded tensor from {filename} has unexpected shape: {tensor.shape}')
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', transform=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure the directory is correct')
        logging.info(f'Creating dataset with {len(self.ids)} images')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
    
    
    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC) #resize
        img = np.asarray(pil_img)
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.transform:
            img, mask = self.transform(img, mask)

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, transform=None):
        super().__init__(images_dir, mask_dir, scale, transform=transform)
