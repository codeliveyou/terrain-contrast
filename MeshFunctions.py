import numpy as np
# import open3d as o3d
import pyvista as pv

def GetMeshByPyvista(data):

    points = np.array(data)

    point_cloud = pv.PolyData(points)

    surface = point_cloud.delaunay_2d()

    return surface


# def GetMeshByOpen3d(data):

#     points = np.array(data)

#     pcd = o3d.geometry.PointCloud()

#     pcd.points = o3d.utility.Vector3dVector(points)

#     pcd.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
#     )

#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

#     o3d.visualization.draw_geometries([mesh])

#     def filter_mesh_by_density(mesh, densities, density_threshold=0.01):
#         vertices_to_remove = densities < np.quantile(densities, density_threshold)
#         mesh.remove_vertices_by_mask(vertices_to_remove)
#         return mesh

#     filtered_mesh = filter_mesh_by_density(mesh, densities, 0.01)

#     return filtered_mesh

