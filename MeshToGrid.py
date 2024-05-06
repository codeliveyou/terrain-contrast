
def convert_3d_cluster_into_height_map(mesh, x_min, x_max, y_min, y_max):
    resolution = 50
    x_range = np.linspace(x_min, x_max, num=resolution)
    y_range = np.linspace(y_min, y_max, num=resolution)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    xy_points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    
    points = np.hstack([xy_points, np.zeros((xy_points.shape[0], 1))])
    distances = mesh.nearest.signed_distance(points)
    height_map = distances.reshape(x_grid.shape)
    return height_map