import open3d as o3d
import numpy as np
from pathlib import Path

run_dir = Path("data/runs/2026-04-21_091522/position_clouds")
for ply in sorted(run_dir.glob("pos_*.ply")):
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
          f"z_range=[{bbox_min[2]:+.3f}, {bbox_max[2]:+.3f}]")