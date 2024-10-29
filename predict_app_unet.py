from flask import Flask
import torch
import asyncio
import websockets
import cv2
import torch
import numpy as np
import threading
from unet import UNet
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import logging
import torch.nn as nn
import torchvision.models.segmentation as segmentation
import time
from pathlib import Path
import concurrent.futures


executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

#Class colors to use in the segmentation
class_colors = {
    0: [0, 0, 0],   #Background(black)
    1: [0, 0, 255], #Bone(red)
    2: [0, 255, 0]  #Ligament(green)
}

model=None
device=None

#Apply the custom color
def apply_custom_color_mapping(mask, class_colors):
    colored_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8) 
    for class_idx, color in class_colors.items():
        colored_mask[mask[0] == class_idx] = color  #Applying color based on prediction class
    return colored_mask

def load_model():
    global model, device
    start_time=time.time()
    dir_model = os.path.expanduser("path_to_the_trained_UNet_model")
    model_path = Path(dir_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    state_dict = torch.load(model_path, map_location=device)
    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)

    #Pre-warm the model
    print("Warming up the model...")

    model.eval()

    for _ in range(5):  #Warm up with 5 dummy images
        dummy_tensor = torch.randn(1, 1, 388, 540).to(device)  #Shape: (1, 3, 388, 540) if the image is 3-color channel
        with torch.no_grad():
            _ = model(dummy_tensor)

    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.4f} seconds.")

async def handle_websocket(websocket, path, img_scale=0.5):
    print("New connection established.")
    try:
        while True:  #Continuous loop to keep the connection open
            try:
                #Wait indefinitely for a message from the app
                message = await websocket.recv() 
                process_time_start = time.time()
                nparr = np.frombuffer(message, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                #Resize and convert to grayscale
                height, width = frame.shape[:2]
                frame_resized = cv2.resize(frame, (int(width * img_scale), int(height * img_scale))) #resize the receive frames
                frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY) #convert the 3-channel to grayscale
              
                #Process the image
                future = executor.submit(process_frame, frame_gray, frame)  #Pass original frame for overlay
                processed_image = await asyncio.wrap_future(future)

                process_time = time.time() - process_time_start
                print(f"Processing in {process_time:.4f} seconds.")

                #Send the processed image back to the App
                await websocket.send(processed_image)

            except websockets.exceptions.ConnectionClosed as e:
                #Handle connection closure
                print(f"Connection closed by client: {e}")
                break

    except Exception as e:
        print(f"Error during processing: {e}")
        await websocket.send("Error during image processing".encode())

def process_frame(frame_gray, original_frame): #predict the images
    frame_tensor = transforms.ToTensor()(frame_gray).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device.type if device.type != 'mps' else 'cpu'):
        pred_mask = model(frame_tensor).argmax(dim=1).cpu().numpy()
    colored_overlay = apply_custom_color_mapping(pred_mask, class_colors)

    colored_overlay_resized = cv2.resize(colored_overlay, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    #Combine the original frame with the overlay
    combined_image = cv2.addWeighted(original_frame, 0.6, colored_overlay_resized, 0.4, 0)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, overlay = cv2.imencode('.jpg', combined_image, encode_param)

    return overlay.tobytes()


async def run_websocket():
    start_time=time.time() #the port intended to be used (in this case "12345") must be changed
      async with websockets.serve(handle_websocket, "0.0.0.0", 12345, ping_interval=60, ping_timeout=120):
        connection_time=time.time()-start_time
        print("Server started on ws://0.0.0.0:12345")
        print(f"Connection completed in {connection_time:.4f} seconds")
        await asyncio.Future()

if __name__ == "__main__":
    start_time=time.time()
    load_model()
    asyncio.run(run_websocket())
    end_time=time.time()-start_time
    print(f"COMPLETION IN {end_time:.4f} seconds")


