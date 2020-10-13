import glob
import os

import numpy as np

from skimage import measure
import tqdm

import open3d as o3d

import torch

from training import build_network
from train import normalize_data, get_prior_z
from utils import dataset


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

    # hyper-parameters
    checkpoint = 'final'
    split = 'test'
    partial_input = True
    z_dim = 256
    nb_grid = 128
    conditioned_ind1 = 1

    save_fold = '/exp_ae/shapenet_car_zdim_256_no_geoinitil'
    try:
        volume = np.load('sdf' + save_fold + '/sdf_{}_{}.npy'.format(checkpoint, conditioned_ind1))
    except FileNotFoundError:
        volume = None

    if volume is None:
        DATA_PATH = 'data/ShapeNet'
        fields = {'inputs': dataset.PointCloudField('pointcloud.npz')}
        test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=['02958343'],
                                               split=split, partial_input=partial_input, evaluation=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False,
                                                  drop_last=False,
                                                  pin_memory=True)

        conditioned_input = test_dataset.__getitem__(conditioned_ind1)['points'].unsqueeze(0)
        print("object:", conditioned_ind1 + 1, "samples:", conditioned_input.shape[1])

        p0_z = get_prior_z(device, z_dim=z_dim)
        net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim, geo_initial=False)
        net = net.to(device)

        net.load_state_dict(torch.load('models' + save_fold + '/model_{}.pth'.format(checkpoint), map_location='cpu'))

        volume = predict(net, conditioned_input, nb_grid)

    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0.0, spacing=(1.0, -1.0, 1.0),
                                                                   gradient_direction='ascent')

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    os.makedirs('output' + save_fold, exist_ok=True)
    o3d.io.write_triangle_mesh('output' + save_fold + '/mesh_{}_{}_{}.ply'.format(split, checkpoint, conditioned_ind1),
                               mesh)
