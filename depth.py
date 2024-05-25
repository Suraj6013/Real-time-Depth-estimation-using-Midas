import cv2
import torch
import time
import numpy as np
import open3d as o3d

# Load MiDas model for depth estimation
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDas", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transformers to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")
transform = midas_transforms.dpt_transform if model_type == "DPT_Large" or model_type == "DPT_Hybrid" else midas_transforms.small_transform

# Open the video capture from a webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("3D Point Cloud")

# Create a geometry object to store the point cloud
pcd = o3d.geometry.PointCloud()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    start = time.time()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transform
    input_batch = transform(img_rgb).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Apply colormap to depth map
    depth_colored = cv2.applyColorMap((depth_map / depth_map.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Generate 3D point cloud with colors
    point_cloud = []
    colors = []
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            depth = depth_map[y, x]
            if depth > 0:
                # Convert pixel coordinates to 3D space
                z = depth
                x_3d = x
                y_3d = depth_map.shape[0] - y
                point_cloud.append([x_3d, y_3d, z])
                colors.append(img_rgb[y, x])

    point_cloud = np.array(point_cloud)
    colors = np.array(colors)

    # Update point cloud data
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Scale point cloud to increase its size
    pcd.scale(1000.0, center=pcd.get_center())

    # Visualize point cloud in 3D space
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # Display camera feed and depth map
    cv2.imshow('Original', img)
    cv2.imshow('Depth Map', depth_colored)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
