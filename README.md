# SonoGravity

The present repository contains frameworks to train, evaluate and predict/segment grayscale images using U-Net and DeepLabV3 Convolutional Neural Networks (CNNs). Additionally, it contains ultrasound images from a Lumbar Spinal Phantom and Dart files used to build a Flutter App. The App aims to brigde the trained models with real-time ultrasound images collection. For that, to simulate a real-time ultrasound, an java script is also available.
The code presented in this repository was developed and used in the MSc Biomedical Engineering dissertation: "Ultrasound-Guided Lumbar Puncture with AI-Based Image Labelling for Intracranial Pressure Assessment in Spaceflight", which aimed to label Longitudinal Spinal Ultrasound images in real-time to be incorporate in a novel and safer Lumbar Puncture technique to assess the Intracranial Pressure in a microgravity environment onboard the International Space Station.
This README provides guidance to train and predict a set of data.

## How to perform the segmentation:

### Step 1: Upload Dependencies
- CUDA 11.7 and PyTorch 1.13 or later and compatible versions of both CUDA and Pytorch must be installed;
- Further dependencies must be installed by running the following comand line:
```bash
pip install torch torchvision tqdm openpyxl numpy Pillow
```
### Step 2: Upload Your Data
Create a folder. Inside that folder, create two more folders for the images and for its respective Ground-truth masks.
The image and its respective mask **MUST** have exactly the same file name.

In the training process, additional data must be introduced to the validation phase. Therefore, another folder with the same characteristics as the one mentioned before must be created to introduce the validation data inside.

Considering that data from a Phantom and Human volunteers were used in this study, only the phantom data is available in the Data folder. The ultrasound files and respective ground-truth masks are represented in a .pth file. It was considered a total of 14 subjects for the phantom images. Although only one phantom model was used, 14 subjects were considered to the frames which were collected in different days, by different operators and in different angles and probe's positions.

The choice of Training and Validation data was made as a 80%/20% distribution, regarding the "subjects". Therefore, the data from the same subject is not in the Training and Validation simultaneously.

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
Scripts have comments to guide some of the steps mentioned in these guidelines.

The directories with the files must be introduced in the train.py scripts. Also, a directory to save the checkpoint, the worksheet with the validation metrics calculated per epoch, and the predicted images must also be introduced.
The colors used on both the ground-truth and the predicted masks are defined in the train.py as well as on the prediction scripts. Three colors were defined for the three classes used. However, both colors and classes must be changed if the user desires or if a different number of classes will be used.
The checkpoints saved during the training are .pth files, which contain the models' weight updates. They are saved per 5 epochs, but this can be altered in the script.

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
The same adaptations performed in train.py scripts must be also used here. However, in these scripts batch size, optimizers, epoch numbers and learning rate are not needed to be adjusted since the introduced data is going to be predictedo per file and evaluated per file.

The --load command must be used to upload the .pth file with the model's weights saved to be used in the prediction.

Thus,
```bash
python predict.py --load 'the_path_to_model' --amp
```
is the format that must be used.

### Step 5 - Prediction with margin:
Considering the nature of the images used in this research, which are characterized by poor spatial resolution, a tolerance system was applied in the metrics calculation.
In this sense, a tolerance chosen by the user creates a dilated and eroded mask on top of the ground-truth mask. The metrics are calculated with the mask which showed better Dice coefficient for the specific file. 
In the scripts evaluate_w_margin.py, the millimiters intended to be used to dilate and erode masks must be specified. It also be specified the number of pixels per millimiter.

The predict_w_margin.py must be used if the user wants to implement this tolerance. 

In the worksheet with the calculated metrics, the files which used eroded or dilated or the original ground-truth masks are discriminated.
### Step 6 - Fine-Tuning:
Fine-Tuning with data augmentation can be performed by running the fine_tuning.py script.
In this, a set of transformations is introduced to the data.
The following transformations are used in the script. If the user wants to introduce more transformations or modify the present ones, must do it in this section.
```bash
custom_transform = CustomCompose([
    RandomRotation(degrees=20), #range of left random rotation
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #color adjustments
    AddGaussianNoiseNumpy(mean=0, std=0.1) #gaussian noise
])
```
To use the Fine-Tuning, the model previously trained and intended to be fine-tuned must be loaded, just like Step 4. Thus, the following command must be used:
```bash
python fine_tunning.py --load 'the_path_to_model' --amp
```

### Outcome's Example:
In this Figure, the original Ultrasound image (from a phantom), respective ground-truth mask and prediction by DeepLab+ResNet-50 are depicted.

![Example of a DeepLab with ResNet-50 prediction](example_prediction.png)

## How to use the App and segment images in real-time:
In this repository, only the essential files used in the App creation are presented. Not all files that are created when a Flutter project is initiated are exhibited.
In the folder "lib", a main.dart file and a "pages" folder can be found. The app is built in these files. 
### Step 1: Download Flutter and Android Studio 
Both Flutter and Android official websites must be accessed to download Flutter and Android Studio. All instructions can be found and followed in: 
- https://flutter.dev/
- https://developer.android.com/studio

### Step2: Upload Dependencies

All dependencies needed are discriminated in the pubspec.yaml file.
To upload all dependencies in this file, in the directory of the app (where this file is incorporated), the following command must be executed:
```build
flutter pub get
```
Inside this file, if needed, dependencies must be added or deleted. 
In the section 

```build
flutter:
  uses-material-design: true

  assets:
    - assets/probe_butterfly.jpg
    - assets/probe0.png
    - assets/probe.jpg
    - assets/logo.png
```
The assets **must** be modified to the images intended to be added and saved in the "assets" folder that is created when a Flutter project is initiated.

## Step 3: Flutter App

The main.dart file and the "pages" folder contain the crucial content of the app.
- main.dart: Contain the App's pages definition with the: 'Home' and 'Prediction'
- pages: contain the dart files of the three pages with explanatory comments

The Home page acts as the app's cover and directs users to the Prediction page when a button is pressed. On the Prediction page, a WebSocket connection is established between the Flutter app and the server that will perform the predictions. Once the connection is established, frames displayed in real time on a Node.js server are transmitted to the app and then forwarded to the prediction server. The resulting predictions are then sent back and displayed within the app.

Both the servers which are displaying the frames in real-time and predicting them must be introduced in this portion, in the predictions.dart file:
```bash
channel = IOWebSocketChannel.connect('ws://path_to_the_server'); //server which is displaying the Ultrasound frames in real-time
model_channel = IOWebSocketChannel.connect('ws://path_to_server_predictions'); //server which is responsible to apply the DL models trained previously
```

### Step 4: Real-time frames Display
A Node.js environment was used to emulate the actions of an ultrasound probe. Thus, a local server was used to display frames at a real-time rate. 

To download the Node.js, please consult and follow the instructions in:
https://nodejs.org/en

server.js defines the server that displays the ultrasound images in real-time. The directories to the frames, frames per second rate and a server's port **MUST** be changed.

To start the server, the following command line must be executed in the directory of the server.js file:
```bash
node server.js
```
### Step 5: Predition script
Both "predict_app_unet.py" and "predict_app_deepLab.py" scripts contain python code that allows a websocket connection between the Flutter App with the trained U-Net and DeepLab models. Thus, the frames from the Node.js environment, after received by the Flutter app, are sent to these scripts, are predicted and sent back to the app to be displayed. 

These files must be placed in the server where the predictions will be made. The dependecies needed in Step 1 of "How to perform the segmentation", at the beginning of these guidelines, are also essential for these two scripts functioning.

The server intended to be used must be introduced here, in both predict_app_unet.py and predict_app_deepLab.py scripts:

```bash
async with websockets.serve(handle_websocket, "0.0.0.0", 12345, ping_interval=60, ping_timeout=120)
```
Moreover, the directory to the .pth file of the trained U-Net or DeepLab model must be introduced in this scripts.
It is relevant to mentioned that the model loaded is pre-warmed before the ultrasound images from the server are sent from the App. This is performed since the first predictions take usually more time.
So, random images with the dimensions of the frames are created and predicted by the model. Ultrasound frames also can be used for this purpose.

The predict_app_unet.py and predict_app_deepLab.py must be executed in the server, depending on which model is intended to be used.

### Step 6: Running everything together

In the directory of the server.js file, the command:

```bash
node server.js
```
must be performed.

Then,

```bash
nohup python predict_app_unet.py
#or
nohup python predict_app_deepLab.py
```
must be executed.

Finally, in the Android Studio, the Flutter code must be executed on a Windows desktop.

### Outcome's Example:

This is the outcome of running the App in the Windows desktop. The app design was made for the dissertation's purpose as well as the prediction exhibited.
![Example](apppp1.png)


## References:
- "Ultrasound-Guided Lumbar Puncture with AI-Based Image Labelling for Intracranial Pressure Assessment in Spaceflight". Beatriz da Silva Pinheiro Gomes Saragoça, Edson Oliveira, Zita Martins. Master's Thesis. Instituto Superior Técnico.
- U-Net's implementation: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351. https://doi.org/10.1007/978-3-319-24574-4_28
- DeepLab's implementation: Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2018). Rethinking Atrous Convolution for Semantic Image Segmentation Liang-Chieh. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4).
