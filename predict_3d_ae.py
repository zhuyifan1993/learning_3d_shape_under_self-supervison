import os

import numpy as np

from skimage import measure
import tqdm

import open3d as o3d

import torch

from training_ae import build_network
from train_3d_ae import normalize_data


def predict(net, conditioned_object, nb_grid):
    x = np.linspace(-1.5, 1.5, nb_grid)
    y = np.linspace(-1.5, 1.5, nb_grid)
    z = np.linspace(-1.5, 1.5, nb_grid)
    X, Y, Z = np.meshgrid(x, y, z)

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    pts = np.stack((X, Y, Z), axis=1)
    pts = pts.reshape(512, -1, 3)

    val = []
    net.eval()
    for p in tqdm.tqdm(pts):
        v = net(torch.Tensor(p).unsqueeze(0), conditioned_object)
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    val = np.concatenate(val)
    val = val.reshape(nb_grid, nb_grid, nb_grid)
    return val


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    c_dim = 256
    save_fold = '/ae/shapenet_cdim_256_100_object_car'
    # save_fold = '/ae/dpg_cdim_0_gargoyle_sup2'

    data = np.load("shapenet/points_shapenet_32x32x32_train.npy")[10]
    # data = np.load('DGP/gargoyle.npy')[::2]
    data = np.expand_dims(data, axis=0)

    print("object num:", len(data), "samples per object:", data.shape[1])
    data = normalize_data(data)
    conditioned_object = torch.from_numpy(data.astype(np.float32))

    net = build_network(input_dim=3, c_dim=c_dim)
    net.to(device)
    net.load_state_dict(torch.load('./models' + save_fold + '/model_final.pth', map_location='cpu'))

    nb_grid = 128
    volume = predict(net, conditioned_object, nb_grid)
    np.save('3d_sdf.npy', volume)

    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0, spacing=(1.0, -1.0, 1.0),
                                                                   gradient_direction='ascent')

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    os.makedirs('output' + save_fold, exist_ok=True)
    o3d.io.write_triangle_mesh('output' + save_fold + '/mesh_object_10_4000.ply', mesh)
