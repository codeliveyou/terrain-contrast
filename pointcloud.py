import math
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
# import open3d as o3d

def pixel_to_world(u, v, d, scale, R, K, t):
    
    # Convert pixel coordinates to camera coordinates
    p = np.array([u, v, 1])
    X_c = d * np.linalg.inv(K).dot(p)
    X_c = X_c * scale
    # Convert camera coordinates to world coordinates
    X_w = R.dot(X_c) + t
    # X_w = R.dot(X_c)
    # Return the height in world coordinates
    return X_w

# def remove_noise(points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     # o3d.visualization.draw([pcd])
#     ci, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
#     # o3d.visualization.draw([ci])
#     nci = np.asarray(ci.points)
#     # print(len(points), len(nci))
#     return nci

def get_pointcloud(image_array):
    scale = 2000 / 255      
    f = 20                  # camera focal length
    width = 304             # image width
    height = 228            # image height
    sensor_x = 36           # camera sensor width
    sensor_y = 24           # camera sensor height
    f_x = width / sensor_x * f
    f_y = height / sensor_y * f
    c_x = width / 2
    c_y = height / 2
    t_x = 0
    t_y = -200
    t_z = 0
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])  # Adjust f_x, f_y, c_x, c_y
    roll, pitch, yaw = -60*np.pi/180, 222*np.pi/180 , 0  # Example values, replace with actual values
    # roll, pitch, yaw = 0,0,0
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    R = r.as_matrix()
    t = np.array([t_x, t_y, t_z])  # Actual camera position


    x = []
    y = []
    z = []

    # final_depth = np.array(image)

    for v in range(image_array.shape[0]):
        for u in range(image_array.shape[1]):
        # u, v = 0, 0  # Pixel coordinates
            d = image_array[v, u]  # Depth at pixel (u, v)
            if d < 50 or d >= 150:
                continue
            X_w = pixel_to_world(u, v, d, scale, R, K, t)
            x.append(X_w[0])
            y.append(X_w[2])
            z.append(-X_w[1])
          
    x, y, z = np.array(x), np.array(y), np.array(z)
    
    return x,y,z
    point_array = np.column_stack((x, y, z))
    point_array = remove_noise(point_array)
    return point_array

if __name__ == '__main__':
    print ("Hello World!")