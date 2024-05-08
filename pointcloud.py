import math
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import open3d as o3d
import random

def pixel_to_world(u, v, d, scale, K, R, t):
    
    # Convert pixel coordinates to camera coordinates
    p = np.array([u, v, 1])
    X_c = d * np.linalg.inv(K).dot(p)
    X_c = X_c * scale
    # Convert camera coordinates to world coordinates
    X_w = R.dot(X_c) + t
    # X_w = R.dot(X_c)
    # Return the height in world coordinates
    return X_w

def remove_noise(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw([pcd])
    ci, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
    # o3d.visualization.draw([ci])
    nci = np.asarray(ci.points)
    # print(len(points), len(nci))
    return nci

def depth_to_points(image, camera_height, angle):
    # Camera parameters
    scale = 1800 / 255      
    f = 22                  # camera focal length
    width = 304             # image width
    height = 228            # image height
    sensor_x = 32         # camera sensor width
    sensor_y = 18           # camera sensor height
    f_x = width / sensor_x * f
    f_y = height / sensor_y * f
    c_x = width / 2
    c_y = height / 2
    t_x = 0
    t_y = -camera_height
    t_z = 0
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])  # Adjust f_x, f_y, c_x, c_y
    roll, pitch, yaw = angle  # Example values, replace with actual values
    # roll, pitch, yaw = 0,0,0
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    R = r.as_matrix()
    t = np.array([t_x, t_y, t_z])  # Actual camera position

    x = []
    y = []
    z = []

    # kernel = np.ones((5,5), np.uint8)
    resized_image = cv2.resize(image, (304, 228))
    # image_eroded = cv2.erode(resized_image, kernel, iterations=1)
    final_depth = np.array(resized_image)

    for v in range(final_depth.shape[0]):
        for u in range(final_depth.shape[1]):
        # u, v = 0, 0  # Pixel coordinates
            d = final_depth[v, u]  # Depth at pixel (u, v)
            if d < 50 or d >= 150:
                continue
            X_w = pixel_to_world(u, v, d, scale, K, R, t)
            if random.randint(0, 4) == 0:
                x.append(X_w[0])
                y.append(X_w[2])
                z.append(-X_w[1])
    
    x, y, z = np.array(x), np.array(y), np.array(z)
    point_array = np.column_stack((x, y, z))
    point_array = remove_noise(point_array)
    return point_array
    # np.save('points.npy', point_array)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    # cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    # cbar.set_label('Z value')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()