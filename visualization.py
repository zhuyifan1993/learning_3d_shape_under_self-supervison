import matplotlib
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
    idx = 3
    data_completeness = 0.7
    data_sparsity = 100
    data = np.load(os.path.join(DATA_PATH, "02958343", model[idx], 'pointcloud.npz'))['points']
    data = create_partial_data(data, idx, data_completeness=data_completeness)[::data_sparsity]

    idx = 1
    data1 = np.load(os.path.join(DATA_PATH, "02958343", model[idx], 'pointcloud.npz'))['points']
    data1 = create_partial_data(data1, idx, data_completeness=data_completeness)[::data_sparsity]

    # plot_pcds(filename=None, pcds=[data], titles=[''])
    # plot_pcds_patterns(filename=None, pcds=[data], titles='t')

    # KITTI dataset
    DATA_PATH = 'data/KITTI-360/data_3d_pointcloud/2013_05_28_drive_0000_sync/car'
    data = np.load(os.path.join(DATA_PATH, '1_canonical.npy'))

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
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    xx = np.linspace(-0.5, 0.5, 128)
    yy = np.linspace(-0.5, 0.5, 128)
    X, Y = np.meshgrid(xx, yy)
    plt.contourf(X, Y, v)
    C = plt.contour(X, Y, v)
    plt.clabel(C, inline=True, fontsize=12, colors='w')
    # plt.scatter(x[:, 0], x[:, 1], color='r')
    plt.show()


def visua_kitti():
    cloud = o3d.io.read_point_cloud("000002_000385.ply")  # Read the point cloud
    o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud


def plot_pcds(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
              xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


def plot_pcds_patterns(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                       xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
    # sdf()
    # visua_kitti()

    a = np.linspace(-0.02, 0.02, 100)
    # relu
    act_relu = torch.nn.ReLU()
    b = np.asarray(act_relu(torch.from_numpy(a)))

    # softplus
    # def softplus(beta):
    #     return lambda x: np.log(1 + np.exp(x * beta)) / beta
    #
    # softp100 = softplus(1)
    # c = softp100(a)

    act_softplus1 = torch.nn.Softplus(beta=1)
    act_softplus100 = torch.nn.Softplus(beta=100)

    c1 = np.asarray(act_softplus1(torch.from_numpy(a)))
    c100 = np.asarray(act_softplus100(torch.from_numpy(a)))

    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(131)
    ax.plot(a, b)
    ax.set_ylabel('output')
    ax.set_title('ReLU')
    ax = fig.add_subplot(132)
    ax.plot(a, c1)
    ax.set_xlabel('input')
    ax.set_title('Softplus(beta=1)')
    ax = fig.add_subplot(133)
    ax.plot(a, c100)
    ax.set_title('Softplus(beta=100)')
    plt.suptitle('Activation Function Behaviour')

    plt.show()
