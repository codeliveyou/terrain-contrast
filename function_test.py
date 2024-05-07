import os
from SARPN.inf import load_sarpn_model, load_sarpn_transform, get_sarpn_output
from pointcloud import get_pointcloud

import matplotlib.pyplot as plt
scale = (604, 456)
model = load_sarpn_model('checkpoint/SARPN_checkpoints_10.pth.tar')
transform = load_sarpn_transform(scale)
depth_array = get_sarpn_output(model, transform, 'sample_input.png', scale)
# point_cloud = get_pointcloud(depth_array)
x,y,z=get_pointcloud(depth_array)


################################ BEGIN XYZ POINT CLOUD TEST ################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
# cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
# cbar.set_label('Z value')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()
################################ END XYZ POINT CLOUD TEST ##################


# plt.imshow(depth_array, cmap='gray')  # 'gray' colormap: black is 0, white is 1
# plt.colorbar()  # Optionally add a colorbar to show the mapping from data values to colors
# plt.title('2D Numpy Array Visualization')
# plt.show()