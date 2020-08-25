import numpy as np
from open3d import *


def main():
    """
    file: points_shapenet_32x32x32_train.npy
    object num: 3122
    each object with 3000 points

    category: car:[0:1191], bottle:[1192:1589]

    """
    data = np.load("shapenet/points_shapenet_32x32x32_train.npy")
    print("object num:", len(data))
    np.savetxt('shapenet/scene1.txt', data[50])
    pcd = read_point_cloud('shapenet/scene1.txt', format='xyz')
    print(pcd)
    draw_geometries([pcd])


if __name__ == "__main__":
    main()
