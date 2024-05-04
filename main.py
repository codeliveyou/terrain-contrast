from PIL import Image
import numpy as np
import pyvista as pv
import open3d as o3d

def get_mesh(data):
    points = np.array(data)

    print(points.shape)

    
    point_cloud = pv.PolyData(points)

    
    surface = point_cloud.delaunay_2d()

    
    surface.plot(show_edges=True)


def get_mesh2(data):
    points = np.array(data)

    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
    )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    o3d.visualization.draw_geometries([mesh])

    def filter_mesh_by_density(mesh, densities, density_threshold=0.01):
        vertices_to_remove = densities < np.quantile(densities, density_threshold)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        return mesh

    filtered_mesh = filter_mesh_by_density(mesh, densities, 0.01)

    o3d.visualization.draw_geometries([filtered_mesh])


def convert_data_into_grayimage(total_data):
    # total_data = [(int(y) for y in x.split(' ')) for x in total_data.split("\n")]

    data = []
    for x in total_data.split("\n"):
        if len(x) < 1:
            break
        temp_array = []
        for y in x.split('	'):
            if len(y) > 0 and y != 'T':
                temp_array.append(float(y))
        # temp_array.append(len(data))
        data.append(temp_array.copy())

    # get_mesh(data)

    data = sorted(data, key = lambda x : x[2])

    mn_height = data[0][2]
    mx_height = data[-1][2]

    data = sorted(data, key = lambda x : x[0])

    final_data = []
    for i in range(100):
        temp_array = sorted(data[i * 100 : (i + 1) * 100], key = lambda x : x[1])
        final_data.append([y[2] for y in temp_array])
    
    return final_data

def value_to_color(value):
    return (max_height - value) / (max_height - min_height)

f = open('data.txt')

f.readline()

total_data = f.read()

final_data = convert_data_into_grayimage(total_data)

min_height = np.min(final_data)
max_height = np.max(final_data)

# print(final_data)


for i in range(100):
    for j in range(100):
        final_data[i][j] = value_to_color(final_data[i][j])

extended_data = []

# print(len(extended_data))

for _ in range(5):
    for i in range(100):
        extended_data.append([])
        for __ in range(5):
            extended_data[i + _ * 100] += final_data[i]


for i in range(500):
    for j in range(500):
        extended_data[i][j] = final_data[i // 5][j // 5]


test_mesh = []

for i in range(500):
    for j in range(500):
        if i % 5 > 0 or j % 5 > 0:
            continue
        test_mesh.append([i, j, extended_data[i][j] * 100])

get_mesh(test_mesh)


data = np.array(extended_data)

scaled_data = (255 * data).astype(np.uint8)

gray_image = Image.fromarray(scaled_data)

gray_image.show()

gray_image.save("gray_image.jpg")

