import sys
import os
import numpy as np
import open3d as o3d
from superquadric import SuperquadricObject
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import plyfile
import yaml

root_dir = '/home/louis/Git/ycb-tools/models/ycb'
model_dir = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]
STEP = 50

with open(f"ycb_sq_params_new.yaml") as file:
    try:
        params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

for d in model_dir:
    p_temp = os.path.join(d, 'google_16k', 'nontextured.ply')
    if not os.path.exists(p_temp):
        p_temp = os.path.join(d, 'tsdf', 'nontextured.ply')
    if not os.path.exists(p_temp):
        p_temp = os.path.join(d, 'clouds', 'merged_cloud.ply')
    if not os.path.exists(p_temp):
        # for ._025_mug
        continue

    model_name = d.split('/')[-1]
    # if not "lock" in model_name:
    #     continue
    print("""""")
    print(model_name, p_temp)

    pcd = o3d.io.read_point_cloud(p_temp) # Read the point cloud
    point_cloud_in_numpy = np.asarray(pcd.points)

    shape = params[model_name]['eps']
    scale = params[model_name]['radii']
    quat = params[model_name]['quat']
    pos = params[model_name]['pos']

    print(shape)
    s1 = SuperquadricObject(*scale, *shape, pos=pos, quat=quat)
    ax = plt.subplot(111, projection='3d')
    s1.plot_sq(plot_type='3D', ax=ax, colour="green")
    ax.scatter(point_cloud_in_numpy[::STEP, 0], point_cloud_in_numpy[::STEP, 1], point_cloud_in_numpy[::STEP, 2])
    ax.set_aspect('equal')

    plt.show()
