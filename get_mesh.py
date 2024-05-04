import numpy as np
import pyvista as pv

# Example: Random points in 3D space (Replace this with your point cloud data)
num_points = 100
np.random.seed(42)
points = np.random.rand(num_points, 3) * 100

# Create a PyVista point cloud object
point_cloud = pv.PolyData(points)

# Apply Delaunay 2D triangulation to the points (assuming all points are on a surface)
surface = point_cloud.delaunay_2d()

# Visualize the surface mesh
surface.plot(show_edges=True)



# import numpy as np
# import open3d as o3d

# # Example: Random points in 3D space (Replace this with your point cloud data)
# num_points = 100
# np.random.seed(42)
# points = np.random.rand(num_points, 3) * 100

# # Create an Open3D point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # Compute the normal vectors of the point cloud (necessary for Poisson Reconstruction)
# pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30)
# )

# # Apply Poisson surface reconstruction
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# # Visualize the mesh
# o3d.visualization.draw_geometries([mesh])

# # Optional: Filter by density (requires manual processing if needed)
# # This example shows how you might do it, but this step is generally optional
# def filter_mesh_by_density(mesh, densities, density_threshold=0.01):
#     vertices_to_remove = densities < np.quantile(densities, density_threshold)
#     mesh.remove_vertices_by_mask(vertices_to_remove)
#     return mesh

# # Apply the optional density filter
# filtered_mesh = filter_mesh_by_density(mesh, densities, 0.01)

# # Visualize the filtered mesh
# o3d.visualization.draw_geometries([filtered_mesh])

