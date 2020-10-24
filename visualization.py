import numpy as np
import os

import torch
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def main():
    """
    file: points_shapenet_32x32x32_train.npy
    object num: 3122
    each object with 3000 points

    category: car:[0:1191], bottle:[1192:1589]

    folder: shapenet_pointcloud
    object num: 783
    each object with 250000 points/normals

    category: car: 0000.npz - 0298.npz
    category: bottle: 0299.npz - 0398.npz
    category: sofa: 0399.npz - 0782.npz

    """
    DATA_PATH = 'data/ShapeNet'
    split_file = os.path.join(DATA_PATH, "02958343", 'test.lst')
    with open(split_file, 'r') as f:
        model = f.read().split('\n')
    idx = 769
    data_completeness = 0.5
    data_sparsity = 200
    data = np.load(os.path.join(DATA_PATH, "02958343", model[idx], 'pointcloud.npz'))['points']
    data = create_partial_data(data, idx, data_completeness=data_completeness)[::data_sparsity]

    np.savetxt('shapenet/scene1.txt', data)
    pcd = o3d.io.read_point_cloud('shapenet/scene1.txt', format='xyz')
    print(pcd)
    o3d.visualization.draw_geometries([pcd])


def create_partial_data(input_data=None, idx=0, data_completeness=0.5):
    data = input_data
    rs = np.random.RandomState(idx)
    offset = rs.rand() - 0.5
    r = R.from_euler('zxy', R.random(num=1, random_state=idx).as_euler('zxy', degrees=True), degrees=True)
    data_rot = r.apply(data)
    ind = np.where(data_rot[:, 1] > offset)
    selected = len(ind[0]) / len(data)
    while selected > data_completeness or selected < data_completeness - 0.1:
        idx += 1
        rs = np.random.RandomState(idx)
        offset = rs.rand() - 0.5
        ind = np.where(data_rot[:, 1] > offset)
        selected = len(ind[0]) / len(data)
    data = input_data[ind]
    return data


def sdf():
    data = np.load('3d_sdf.npy')
    v = data[:, :, 64]
    plt.figure(figsize=(6, 6))
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])

    xx = np.linspace(-1.5, 1.5, 128)
    yy = np.linspace(-1.5, 1.5, 128)
    X, Y = np.meshgrid(xx, yy)
    plt.contourf(X, Y, v)
    C = plt.contour(X, Y, v)
    plt.clabel(C, inline=True, fontsize=12, colors='w')
    # plt.scatter(x[:, 0], x[:, 1], color='r')
    plt.show()


def visua_kitti():
    cloud = o3d.io.read_point_cloud("000002_000385.ply")  # Read the point cloud
    o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud


if __name__ == "__main__":
    main()
    # sdf()
    # visua_kitti()
