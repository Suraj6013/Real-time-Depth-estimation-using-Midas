import cv2
import torch
import time
import numpy as np



# Load MiDas model for depth estimation
model_type = "DPT_Hybrid"
#model_type = "MiDas_small"

midas = torch.hub.load("intel-isl/MiDas", model_type)

# move model to GPU if available
device = torch.device("cuda") if torch .cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# loadd transformers to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    tranform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# open up the vedio capture from a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    #apply input transform
    input_batch = tranform(img).to(device)

    # prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,               
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'fps: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    if cv2.waitKey(5) & 0xFF ==27:
        break

cap.release()