import glob
import os

import numpy as np

from skimage import measure
import tqdm

import open3d as o3d

import torch

from training_vae import build_network
from train_3d_vae import normalize_data, get_prior_z


def predict(net, conditioned_input, nb_grid):
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
    latent_code, _ = net.encoder(conditioned_input)
    # print(latent_code)
    latent_code = latent_code.unsqueeze(dim=1).expand((-1, pts.shape[1], -1))
    for p in tqdm.tqdm(pts):
        v = net.decoder(torch.Tensor(p).unsqueeze(0), latent_code)
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    val = np.concatenate(val)
    val = val.reshape(nb_grid, nb_grid, nb_grid)
    return val


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    z_dim = 256
    save_fold = '/exp/shapenet_car_zdim_256_partial'

    DATA_PATH = 'data/ShapeNet'
    split_file = os.path.join(DATA_PATH, "02958343", 'test.lst')
    with open(split_file, 'r') as f:
        model = f.read().split('\n')
    data = np.load(os.path.join(DATA_PATH, "02958343", model[0], 'pointcloud.npz'))['points']
    ind = np.where(np.logical_or(np.logical_or(data[:, 0] < 0, data[:, 1] < 0), data[:, 2] < 0))
    data = data[ind]
    data = np.expand_dims(data, axis=0)

    print("object num:", len(data), "samples per object:", data.shape[1])
    data = normalize_data(data)
    conditioned_input = torch.from_numpy(data.astype(np.float32))

    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim)
    # net = torch.nn.DataParallel(net).to(device)

    net.load_state_dict(torch.load('./models' + save_fold + '/model_0100.pth', map_location='cpu'))

    nb_grid = 128
    volume = predict(net, conditioned_input, nb_grid)

    # volume = np.load('models' + save_fold + '/3d_sdf_0500.npy')

    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0, spacing=(1.0, -1.0, 1.0),
                                                                   gradient_direction='ascent')

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    os.makedirs('output' + save_fold, exist_ok=True)
    o3d.io.write_triangle_mesh('output' + save_fold + '/mesh_car_test0_100.ply', mesh)
