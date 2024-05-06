from PIL import Image
import matplotlib.pyplot as plt
import trimesh
from scipy.interpolate import griddata

from MeshFunctions import *
from HOG import *

def show_tri_mesh(tri_mesh):
    # print("Number of cells in the mesh:", tri_mesh.vertices)
    # print("Number of strict faces in the mesh:", tri_mesh.faces)
    plotter = pv.Plotter()
    plotter.add_mesh(tri_mesh, show_edges=True, color='lightblue')
    plotter.add_axes()
    plotter.show()

def convert_data_into_2d_matrix(total_data, height, width):

    data = []
    for x in total_data.split("\n"):
        if len(x) < 1:
            continue
        temp_array = []
        for y in x.split('	'):
            temp_array.append(float(y))
        data.append(temp_array.copy())

    data = sorted(data, key = lambda x : -x[0])

    final_data = []
    for i in range(height):
        temp_array = sorted(data[i * width : (i + 1) * width], key = lambda x : x[1])
        final_data.append([y[2] for y in temp_array])

    return final_data

def value_to_color(value, min_height, max_height):
    return (max_height - value) / (max_height - min_height)

def show_3d_cluster(p):

    points = np.array(p)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x, y, z, s=1)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def get_submatrix(matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]

def show_gray_image(image_gray):
    plt.imshow(image_gray, cmap='gray')
    plt.colorbar()
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()

def mesh_into_height_map(mesh, x_min, x_max, y_min, y_max, x_resolution, y_resolution):
    vertices = mesh.vertices
    x_range = np.linspace(x_min, x_max, num=x_resolution)
    y_range = np.linspace(y_min, y_max, num=y_resolution)

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_values = griddata(vertices[:, :2], vertices[:, 2], (x_grid, y_grid), method='linear')
    z_values_filled = np.nan_to_num(z_values, nan=np.nanmin(z_values))

    return [list(z_values_filled[i]) for i in range(len(z_values_filled) - 1, -1, -1)]

def pv_mesh_into_tri_mesh(pv_mesh):
    faces_as_array = pv_mesh.faces.reshape((pv_mesh.n_cells, 4))[:, 1:]
    tmesh = trimesh.Trimesh(pv_mesh.points, faces_as_array)
    return tmesh

def read_cluster():
    target_x = np.load('./Target Data/x.npy')
    target_y = np.load('./Target Data/y.npy')
    target_z = np.load('./Target Data/z.npy')
    target_len = len(target_x)

    target_points = []

    for i in range(target_len):
        target_points.append([target_x[i], target_y[i], target_z[i]])
    
    return target_points

def read_map_data(file_name, height, width):
    f = open(file_name)
    total_data = f.read()
    return convert_data_into_2d_matrix(total_data, height, width)
   
def height_map_into_gray_image(map_data, min_height, max_height):
    gray_image = map_data.copy()
    for i in range(len(map_data)):
        for j in range(len(map_data[i])):
            gray_image[i][j] = value_to_color(map_data[i][j], min_height, max_height)

    data = np.array(gray_image)
    scaled_data = (255 * data).astype(np.uint8)
    scaled_data.setflags(write=1)
    img = Image.fromarray(scaled_data, 'L')
    return img

def point_cluster_into_height_map(points):
    mesh = GetMeshByPyvista(points)
    tri_mesh = pv_mesh_into_tri_mesh(mesh)
    # show_3d_cluster(points) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # mesh.plot(show_edges=True) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)
    x_resolution = int(round((x_max - x_min) / 5))
    y_resolution = int(round((y_max - y_min) / 5))
    
    target_height_map = mesh_into_height_map(tri_mesh, x_min, x_max, y_min, y_max, x_resolution, y_resolution)
    return target_height_map

def confident_score(imageA, imageB):
    max_score = 0
    height_A = imageA.shape[0]
    width_A = imageA.shape[1]

    height_B = imageB.shape[0]
    width_B = imageB.shape[1]

    target_X = 0
    target_Y = 0

    for coner_X in range(0, height_A - height_B + 1, 200):
        for coner_Y in range(0, width_A - width_B + 1, 200):
            current_correlation = compare_hog_features(imageA[coner_X : coner_X + height_B, coner_Y : coner_Y + width_B], imageB)
            # print(current_correlation)
            if current_correlation > max_score:
                target_X = coner_X
                target_Y = coner_Y
                max_score = current_correlation
    # show_image(imageA[target_X : target_X + height_B, target_Y : target_Y + width_B])

    for x in range(target_X, target_X + height_B):
        for i in range(5):
            imageA[x][target_Y + i - 2] = 0
            imageA[x][target_Y + width_B + i - 2] = 0
    for y in range(target_Y, target_Y + width_B):
        for i in range(5):
            imageA[target_X + i - 2][y] = 0
            imageA[target_X + height_B + i - 2][y] = 0

    show_image(imageA) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return [max_score, target_X, target_Y]

def main():
    # Map Part
    height, width = 1000, 1600
    map_data = read_map_data("./Map Data/data.txt", height, width)

    min_height = np.min(map_data)
    max_height = np.max(map_data)

    # Target Part
    target_points = read_cluster()
    target_map = point_cluster_into_height_map(target_points)
    t_min_height = np.min(target_map)
    t_max_height = np.max(target_map)

    # print(t_min_height, t_max_height)

    # Comparing part
    min_height = min(min_height, t_min_height)
    max_height = max(max_height, t_max_height)

    map_gray_image = height_map_into_gray_image(map_data, min_height, max_height)
    t_gray_image = height_map_into_gray_image(target_map, min_height, max_height)

    # map_gray_image.show() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # t_gray_image.show() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    t_gray_image.save("./Images/t_gray_image.jpg")
    map_gray_image.save("./Images/map_gray_image.jpg")

    imageA = cv2.imread("./Images/map_gray_image.jpg")
    imageB = cv2.imread("./Images/t_gray_image.jpg")
    max_score, target_X, target_Y = confident_score(imageA, imageB)
    print("Maximum confident score is : " + str(max_score), "Target position is ", target_X, target_Y)

main()
