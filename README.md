# SonoGravity

The present repository contain frameworks to train, evaluate and predict/segment grayscale images using U-Net and DeepLabV3 Convolutional Neural Networks (CNNs).
The code presented in this repository was developed and used in the MSc Biomedical Engineering dissertation: "Ultrasound-Guided Lumbar Puncture with AI-Based Image Labelling for Intracranial Pressure Assessment in Spaceflight", which aimed to label Longitudinal Spinal Ultrasound images to be incorporate in a novel and safer Lumbar Puncture technique to assess the Intracranial Pressure in a microgravity environment onboard of the International Space Station.
This README provides guidance to train and predict a set of data.

## How to train U-Net and DeepLab models with ResNet-50 and ResNet-101 backbones:

### Step 1: Upload Necessary Dependencies


### Step 2: Upload Your Data
In a folder, create two folders for the images and for its respective Ground-truth masks.
The image and its respective mask **MUST** have exactly the same file name.

In the training process, additional data must be introduced to the validation phase. Therefore, another folder with the same characteristics as the one mentioned before must be created to introduce the validation data inside

### Step 3: Frameworks adaptation
The code is adapted to greyscale data, therefore if the data aimed to be segmented has a 3-color channel, scripts must be modified, specifically in the lines:
- U-Net:
  
This portion 
```bash
model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
```
must be modified to:
```bash
model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
```
- DeepLab:
  
This portion
```bash
model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```
must be modified to
```bash
model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```
or simply deleted, since DeepLabV3 Pytorch accepts data with 3-color channels by default.

### Step 3: Train

Both scripts train.py inside U-Net and DeepLab folders must be used in the training process.
Scripts have comments to guide some of the steps mentioned in this guide.

The directories with the files must be introduced in the train.py scripts. Also, a directories to save the checkpoint, the worksheet with the validation metrics calculated per epoch, and the predicted images must also be introduced.
The checkpoints saved during the training are .pth files, which contain the models' weight updates. They are saved per 5 epoch, but this can be altered in the script.

If it is intended to be used DeepLab with ResNet-101 instead of ResNet-50 backbone, the code line:
```bash
model = segmentation.deeplabv3_resnet50(pretrained=False)
```
must be replaced by:
```bash
model=segmentation.deeplabv3_resnet101(pretrained=False) 
```

Parameters as batch size, L2 regularization, learning rate, optimizer type, number of epochs, number of classes intended to predict can be changed in the script.
However, the batch size, number of epochs and classes can be defined in the command line as well.

The command to train both architectures is the following:
```bash
python train.py
```
Depending on the model intended to be used, this command line must be used inside the specific folder.
To implement the parameters in the command line, the following can be used:
- Number of epochs: --epochs E
- Batch size: --batch-size B
- Learning rate: --learning-rate LR
- Scale: --scale SCALE
- Mixed precision: --amp

For example:
```bash
python train.py --epochs 100 --batch-size 8 --amp
```
### Step 4 - Prediction:
To Test the model trained or evaluate the Validation phase per image, the predict.py scripts must be used.
The same adaptations performed in train.py scripts must be also used here. However, in these scripts batch size, optimizers, epoch numbers and learning rate are not needed to be adjusted since the introduced data is going to be predict per file and evaluated per file.

The --load command must be used to upload the .pth file with the model's weights saved to be used in the prediction.

Thus,
```bash
python predict.py --load 'the_path_to_model' --amp
```
is the format that must be used.

### Step 4 - Prediction with margin:
Considering the nature of the images used in this research, which are characterized by poor spatial resolution, a tolerance system was applied in the metrics calculation.
In this sense, a tolerance chose by the user creates a dilated and eroded mask on top of the ground-truth mask. The metrics are calculated with the mask which showed better Dice coefficient for the specific file. 
In the scripts evaluate_w_margin.py, the millimiters intended to be used to dilate and erode masks must specified. It also be specified the number of pixels per millimiter.

The predict_w_margin.py must be used if the user want to use this tolerance. 

In the worksheet with the calculated metrics, the files which used eroded or dilated or the original ground-truth masks are discriminated.

## Outcome's Example:




## References:
- "Ultrasound-Guided Lumbar Puncture with AI-Based Image Labelling for Intracranial Pressure Assessment in Spaceflight". Beatriz da Silva Pinheiro Gomes Saragoça, Edson Oliveira, Zita Martins. Master's Thesis. Instituto Superior Técnico.
- U-Net's implementation: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351. https://doi.org/10.1007/978-3-319-24574-4_28
- DeepLab's implementation: Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). Rethinking Atrous Convolution for Semantic Image Segmentation Liang-Chieh. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4).
