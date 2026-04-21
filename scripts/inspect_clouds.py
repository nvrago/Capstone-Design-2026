import open3d as o3d
import numpy as np
from pathlib import Path
import sys

# absolute path from repo root
run_dir = Path("/home/capstone/Capstone-Design-2026/data/runs/2026-04-21_091522/position_clouds")

print(f"looking in: {run_dir}")
print(f"exists: {run_dir.exists()}")

if not run_dir.exists():
    sys.exit(1)

plys = sorted(run_dir.glob("pos_*.ply"))
print(f"found {len(plys)} ply files\n")

for ply in plys:
    pcd = o3d.io.read_point_cloud(str(ply))
    if len(pcd.points) == 0:
        print(f"{ply.name}: empty")
        continue
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    print(f"{ply.name}: n={len(pts):6d}  "
          f"centroid=[{centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f}]  "
          f"z=[{bbox_min[2]:+.3f}, {bbox_max[2]:+.3f}]  "
          f"x=[{bbox_min[0]:+.3f}, {bbox_max[0]:+.3f}]  "
          f"y=[{bbox_min[1]:+.3f}, {bbox_max[1]:+.3f}]")