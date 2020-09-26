import numpy as np
from open3d import *
from train_3d_ae import normalize_data
import matplotlib.pyplot as plt


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
    # data = np.load("shapenet/points_shapenet_32x32x32_train.npy")[1205, ::]
    data = np.load("shapenet_pointcloud/0000.npz")['points']
    data = np.expand_dims(data, axis=0)
    data = normalize_data(data).squeeze(0)
    # np.savetxt('shapenet/scene1.txt', data[np.argsort(data, axis=0)[:, 0]][:1500:])
    np.savetxt('shapenet/scene1.txt', data)
    pcd = read_point_cloud('shapenet/scene1.txt', format='xyz')
    print(pcd)
    draw_geometries([pcd])


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


if __name__ == "__main__":
    main()
    # sdf()
