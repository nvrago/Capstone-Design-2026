import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("/home/capstone/Capstone-Design-2026/data/reference/dome_cloud.ply")
pts = np.asarray(pcd.points)
print(f"dome: n={len(pts)}")
print(f"  centroid=[{pts.mean(0)[0]:+.3f}, {pts.mean(0)[1]:+.3f}, {pts.mean(0)[2]:+.3f}]")
print(f"  x=[{pts.min(0)[0]:+.3f}, {pts.max(0)[0]:+.3f}]")
print(f"  y=[{pts.min(0)[1]:+.3f}, {pts.max(0)[1]:+.3f}]")
print(f"  z=[{pts.min(0)[2]:+.3f}, {pts.max(0)[2]:+.3f}]")