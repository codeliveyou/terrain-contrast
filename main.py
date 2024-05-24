from PIL import Image
import matplotlib.pyplot as plt
import trimesh
from scipy.interpolate import griddata
import matplotlib.image as mpimg

# from SARPN.inf import load_sarpn_model, load_sarpn_transform, get_sarpn_output
from pointcloud import *

from MeshFunctions import *
from HOG import *
from Classes import *
from Track import earth_R, aircraft

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


    # print("max_N = " + str(max([x[0] for x in data])))
    # print("min_E = " + str(min([x[1] for x in data])))
    # print("min_N = " + str(min([x[0] for x in data])))
    # print("max_E = " + str(max([x[1] for x in data])))
    # print("min_H = " + str(min([x[2] for x in data])))
    # print("max_H = " + str(max([x[2] for x in data])))
    
    final_data = []

    data = sorted(data, key = lambda x : -x[0])
    for i in range(height):
        temp_array = sorted(data[i * width : (i + 1) * width], key = lambda x : x[1])
        final_data.append([y[2] for y in temp_array])
    
    # data = sorted(data, key = lambda x : -x[0])
    # for i in range(width):
    #     temp_array = sorted(data[i * width : (i + 1) * width], key = lambda x : x[1])
    #     mx = max(mx, max([data[x * width + i][1] for x in range(height)]) - min([data[x * width + i][1] for x in range(height)]))
    # print(mx)

    return final_data

def value_to_color(value, min_height, max_height):
    return (max_height - value) / (max_height - min_height)

def show_3d_cluster(points):
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
    gray_image = np.array(map_data)
    for i in range(len(map_data)):
        for j in range(len(map_data[i])):
            gray_image[i][j] = value_to_color(map_data[i][j], min_height, max_height)

    data = np.array(gray_image)
    scaled_data = (255 * data).astype(np.uint8)
    scaled_data.setflags(write=1)
    img = Image.fromarray(scaled_data, 'L')
    return img

def point_cluster_into_height_map(points):
    # show_3d_cluster(points) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # exit(0)
    
    mesh = GetMeshByPyvista(points)
    tri_mesh = pv_mesh_into_tri_mesh(mesh)
    # show_tri_mesh(mesh)#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)

    x_resolution = int(round((x_max - x_min) / 5))
    y_resolution = int(round((y_max - y_min) / 5))
    
    target_height_map = mesh_into_height_map(tri_mesh, x_min, x_max, y_min, y_max, x_resolution, y_resolution)

    return target_height_map, point2(x_min, y_max)

def confident_score(imageA, imageB, ind = 0):
    max_score = 0
    height_A = imageA.shape[0]
    width_A = imageA.shape[1]

    height_B = imageB.shape[0]
    width_B = imageB.shape[1]

    # print(imageA.shape, imageB.shape)
    # exit(0)

    target_X = 0
    target_Y = 0

    for coner_X in range(0, height_A - height_B + 1, 100):
        for coner_Y in range(0, width_A - width_B + 1, 100):
            current_correlation = compare_hog_features(imageA[coner_X : coner_X + height_B, coner_Y : coner_Y + width_B], imageB)
            # print(current_correlation)
            if current_correlation > max_score:
                target_X = coner_X
                target_Y = coner_Y
                max_score = current_correlation
    # show_image(imageA[target_X : target_X + height_B, target_Y : target_Y + width_B])

    for x in range(target_X, target_X + height_B):
        for i in range(5):
            imageA[x][target_Y + i] = 0
            imageA[x][target_Y + width_B - i] = 0
    for y in range(target_Y, target_Y + width_B):
        for i in range(5):
            imageA[target_X + i][y] = 0
            imageA[target_X + height_B - i][y] = 0
    # show_image(imageA) #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    cv2.imwrite(F"./Images/result{ind}.jpg", imageA)
    return [max_score, target_X, target_Y]

def get_target_area(current_N, current_E, azimuth):
    # print(current_N, current_E, azimuth)
    LB = point2().E_from_angle(azimuth + pi / 2) * 1000
    # print('LB: ', LB, point2())
    RB = point2() - LB
    LT = LB - LB.conj() * 2
    RT = RB + RB.conj() * 2
    # LTP = point2(min([LB.x, RB.x, LT.x, RT.x]), max([LB.y, RB.y, LT.y, RT.y]))
    # RBP = point2(max([LB.x, RB.x, LT.x, RT.x]), min([LB.y, RB.y, LT.y, RT.y]))
    RBP = point2(max([0, LT.x, RT.x]), min([0, LT.y, RT.y]))
    LTP = point2(min([0, LT.x, RT.x]), max([0, LT.y, RT.y]))

    # LT_aircraft = aircraft(current_N, current_E)
    # LT_aircraft.set_direction(0, azimuth)
    # LT_aircraft.position = point(LTP.x, LTP.y, 0)
    # LT_aircraft.update_NE()

    # RB_aircraft = aircraft(current_N, current_E)
    # RB_aircraft.set_direction(0, azimuth)
    # RB_aircraft.position = point(RBP.x, RBP.y, 0)
    # RB_aircraft.update_NE()

    LT_N = current_N + (LTP.y / earth_R) / pi * 180
    LT_E = current_E + (LTP.x / (earth_R * math.cos(current_N / 180 * pi))) / pi * 180
    
    RB_N = current_N + (RBP.y / earth_R) / pi * 180
    RB_E = current_E + (RBP.x / (earth_R * math.cos(current_N / 180 * pi))) / pi * 180
    
    return [[LT_N, LT_E], [RB_N, RB_E]]

def modify_part():
    # Map Part
    height, width = 1000, 1600
    map_data = read_map_data("./Map Data/data.txt", height, width)

    min_height = np.min(map_data)
    max_height = np.max(map_data)

    # Target Part
    target_points = read_cluster()
    target_map, _ = point_cluster_into_height_map(target_points)

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

# def get_cluster(model, transform, image_path, current_height, inclination_angle):
#     scale = (604, 456)
#     depth_array  = get_sarpn_output(model, transform, image_path, scale)
#     x,y,z=get_pointcloud(depth_array)
#     lx = list(x)
#     ly = list(y)
#     lz = list(z)
#     return [[lx[i], ly[i], lz[i]] for i in range(len(lx))]

def draw_path(
    a,
    b,
    height=1920,
    width=1080,
    img_path="./Images/map.png",
    min_y=39.1025444444444,
    max_y=39.1479055555556,
    min_x=125.818866666667,
    max_x=125.894291666667,
):
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])

    img = mpimg.imread(img_path)
    plt.imshow(img, extent=[min_x, max_x, min_y, max_y])

    def plot(dots, color):
        for index, dot in enumerate(dots):
            plt.plot(dot[1], dot[0], color)
            if index % 20 == 0:
                plt.text(dot[1], dot[0], str(index), color="yellow")

    plot(a, "bo")
    plot(b, "ro")
    plt.show()
    plt.show()

def test_part():
    # Map Part
    
    # us
    # height, width = 1000, 1600
    # max_N = 39.865277778
    # min_E = -105.333330278
    # min_N = 39.820322778
    # max_E = -105.239611111

    # k_py
    height, width = 900, 1200
    max_N = 39.146888889
    min_E = 125.8225025
    min_N = 39.106433889
    max_E = 125.892222222
    
    map_data = read_map_data("./Map Data/k_py.txt", height, width)
    # exit(0)

    min_height = np.min(map_data)
    max_height = np.max(map_data)

    map_gray_image = height_map_into_gray_image(map_data, min_height, max_height)
    map_gray_image.save("./Images/k_py_gray.jpg")
    # map_gray_image.show()
    # return

    drone_Expected_positions = []
    drone_real_positions = []

    file = open("./Target Data/drone.txt")
    input_drone_data = [x.split(' ') for x in file.readlines()]

    # Target Part
    for i in range(0, 20):
        print(f'INFO: loading {i} image..')
        current_E = float(input_drone_data[i][1])
        current_N = float(input_drone_data[i][0])
        current_H = 600
        inclination_degree = float(input_drone_data[i][3])
        azimuth_degree = float(input_drone_data[i][4])

        # print(current_N, current_E, inclination_degree, azimuth_degree)
        # exit(0)
        
        azimuth = azimuth_degree / 180 * pi
        inclination_angle = inclination_degree / 180 * pi
        # print(azimuth_degree)
        target_LT, target_RB = get_target_area(current_N, current_E, -(azimuth - math.pi / 2))
        # print(target_LT, target_RB)
        # max_N = max(max_N, target_LT[0])
        # min_N = min(min_N, target_RB[0])
        # max_E = max(max_E, target_RB[1])
        # min_E = min(min_E, target_LT[1])
        # continue
        
        LT_x = round((target_LT[0] - max_N) / (min_N - max_N) * height) - 50
        LT_y = round((target_LT[1] - min_E) / (max_E - min_E) * width) - 50
        RB_x = round((target_RB[0] - max_N) / (min_N - max_N) * height) + 50
        RB_y = round((target_RB[1] - min_E) / (max_E - min_E) * width) + 50

        if LT_x < 0:
            RB_x += abs(LT_x)
            LT_x = 0
        if RB_x > height:
            LT_x -= RB_x - height
            RB_x = height
        if LT_y < 0:
            RB_y += abs(LT_y)
            LT_y = 0
        if RB_y > width:
            LT_y -= RB_y - width
            RB_y = width

        # print(min_N, max_N, min_E, max_E)
        # print("------>>        ", target_LT, target_RB)
        # print("------>>        ", LT_x, LT_y, RB_x, RB_y)

        # print(len(target_points), len(target_points[0]))

        # target_points = get_cluster(model, transform, F"./Images/image_{i}.jpg", current_H, inclination_angle)
        depth_image = cv2.imread(F"./Images/gray/SavedImage{i}_Image.png", cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("A", depth_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        target_points = depth_to_points(depth_image, current_H, [-inclination_angle, azimuth, 0.])
        print("Obtained 3D Point cluster")
        # show_3d_cluster(target_points)

        target_map, LT = point_cluster_into_height_map(target_points)
        # exit(0)
        # print(len(target_map), len(target_map[0]))


        # print(LT_x, LT_y, RB_x, RB_y, RB_x - LT_x, len(target_map), RB_y - LT_y, len(target_map[0]))
        # exit(0)
        search_map_data = [map_data[x][LT_y : RB_y] for x in range(LT_x, RB_x)]
        search_map_gray_image = height_map_into_gray_image(search_map_data, min_height, max_height)
        t_gray_image = height_map_into_gray_image(target_map, min_height, max_height)

        # search_map_gray_image.show() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # t_gray_image.show() #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        search_map_gray_image.save(F"./Images/search_map_gray_image{i}.jpg")
        t_gray_image.save(F"./Images/t_gray_image{i}.jpg")

        imageA = cv2.imread(F"./Images/search_map_gray_image{i}.jpg")
        imageB = cv2.imread(F"./Images/t_gray_image{i}.jpg")
        # print(imageA.shape, imageB.shape)
        # exit(0)
        max_score, target_X, target_Y = confident_score(imageA, imageB, i)
        target_X += LT_x
        target_Y += LT_y

        LT_N = max_N - (max_N - min_N) / height * target_X
        LT_E = min_E + (max_E - min_E) / width * target_Y

        drone_N = LT_N - (LT.y / earth_R) * 180 / pi
        drone_E = LT_E - (LT.x / (earth_R * math.cos(LT_N / 180 * pi))) * 180 / pi

        drone_real_positions.append([current_N, current_E])
        drone_Expected_positions.append([drone_N, drone_E])

        print(F"Maximum confident score of image{i} is : " + str(max_score), "Target position is ", F"{drone_N} {drone_E}\n")

    # print(max_N, min_E, min_N, max_E)
    # return
    # print('\n'.join([(str(d[0]) + ',' + str(d[1])) for d in LT_Expected_positions]))
    # print("--------------------------------------------")
    # print('\n'.join([(str(d[0]) + ',' + str(d[1])) for d in drone_real_positions]))
    
    draw_path(drone_real_positions, drone_Expected_positions) #, height, width, min_N, max_N, min_E, max_E, )



def main():
    # modify_part()
    test_part()

main()
