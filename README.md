Real-time Depth Estimation with MiDaS.

This repository contains a Python script for real-time depth estimation using the MiDaS (Monocular Depth Estimation in the Wild with Convolutional Networks) model. The script captures video input from a webcam in real-time, processes each frame through the MiDaS model, and displays the original video along with the corresponding depth map.

Features
MiDaS Depth Estimation: Utilizes the MiDaS model for monocular depth estimation, providing accurate depth maps for each frame of the input video.
Real-time Processing: The script processes webcam input in real-time, allowing users to visualize depth information as the video feed is captured.
FPS Display: The frames per second (FPS) are displayed on the original video feed, providing information about the processing speed.
Color-Mapped Depth Visualization: The depth maps are color-mapped for better visualization, using the 'Magma' colormap.
Usage
Clone the repository: git clone https://github.com/your-username/real-time-depth-estimation.git
Install dependencies: pip install -r requirements.txt
Run the script: python depth_estimation.py
Requirements
Python 3.x
OpenCV
PyTorch
NumPy
Model
The depth estimation model used in this project is the MiDaS model, specifically the DPT Hybrid version. You can find more information about the model here.

Acknowledgments
This project makes use of the MiDaS model developed by Intel-isl. Check out their repository here.
Feel free to contribute, open issues, or use this project as a starting point for your depth estimation applications. If you find it helpful, don't forget to star the repository! 🌟
