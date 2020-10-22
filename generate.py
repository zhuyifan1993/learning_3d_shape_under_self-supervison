import glob
import os
import logging

import numpy as np

import torch
from skimage import measure

import open3d as o3d

from network.training import build_network
from train import get_prior_z
from utils import dataset
import utils.plots as plt


def predict(net, conditioned_input, nb_grid, device, interp=False):
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
    conditioned_input = conditioned_input.to(device)

    if interp:
        latent_z = conditioned_input.unsqueeze(dim=1).expand((-1, pts.shape[1], -1))
        for p in pts:
            v = net.decoder(torch.Tensor(p).unsqueeze(0).to(device), latent_z)
            v = v.reshape(-1).detach().cpu().numpy()
            val.append(v)
    else:
        latent_z, _ = net.encoder(conditioned_input)
        latent_z = latent_z.unsqueeze(dim=1).expand((-1, pts.shape[1], -1))
        for p in pts:
            v = net.decoder(torch.Tensor(p).unsqueeze(0).to(device), latent_z)
            v = v.reshape(-1).detach().cpu().numpy()
            val.append(v)
    val = np.concatenate(val)
    val = val.reshape(nb_grid, nb_grid, nb_grid)

    return val


def interpolation(lambda1, model, img1, img2, device):
    model.eval()
    # latent vector of first input
    img1 = img1.to(device)
    latent_1, _ = model.encoder(img1)

    # latent vector of second input
    img2 = img2.to(device)
    latent_2, _ = model.encoder(img2)

    # interpolation of the two latent vectors
    inter_latent = (1 - lambda1) * latent_1 + lambda1 * latent_2

    return inter_latent


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # hyper-parameters
    checkpoint = '0400'
    partial_input = True
    split = 'test'
    z_dim = 256
    nb_grid = 128
    conditioned_ind1 = 0
    conditioned_ind2 = 769
    eval_fullsque = True
    latentsp_interp = False
    save_mesh = True
    save_pointcloud = True

    save_fold = '/exp_2000/shapenet_car_zdim_256_partial_vae'
    os.makedirs('output' + save_fold, exist_ok=True)

    # build network
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim, geo_initial=False)
    net = net.to(device)
    net.load_state_dict(torch.load('models' + save_fold + '/model_{}.pth'.format(checkpoint), map_location='cpu'))

    # create dataloader
    DATA_PATH = 'data/ShapeNet'
    fields = {'inputs': dataset.PointCloudField('pointcloud.npz')}
    test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=['02958343'],
                                           split=split, partial_input=partial_input, evaluation=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
                                              pin_memory=True)

    if eval_fullsque:
        for ind, data in enumerate(test_loader):
            conditioned_input = data['points']
            print("object:", ind + 1, "samples:", conditioned_input.shape[1])

            net.eval()
            conditioned_input = conditioned_input.to(device)
            latent_code, _ = net.encoder(conditioned_input)

            if not partial_input:
                input_pc = conditioned_input.squeeze()
                all_latent = latent_code.repeat(input_pc.shape[0], 1)
                points = torch.cat([all_latent, input_pc], dim=-1).detach()
                is_uniform = False
            else:
                points = None
                is_uniform = True

            surface = plt.get_surface_trace(points=points, decoder=net.decoder, latent=latent_code, resolution=nb_grid,
                                            mc_value=0, is_uniform=is_uniform, verbose=False, save_ply=True, connected=True)
            if save_mesh:
                surface['mesh_export'].export(
                    'output' + save_fold + '/mesh_{}_{}_{}.off'.format(split, checkpoint, ind), 'off')
            if save_pointcloud:
                surface['mesh_export'].export(
                    'output' + save_fold + '/mesh_{}_{}_{}.ply'.format(split, checkpoint, ind), 'ply')

    # Interpolate in Latent Space
    if latentsp_interp:
        x1 = test_dataset.__getitem__(conditioned_ind1)['points'].unsqueeze(0)
        x2 = test_dataset.__getitem__(conditioned_ind2)['points'].unsqueeze(0)
        lambda_range = np.linspace(0, 1, 10)
        for i in range(len(lambda_range)):
            inter_latent = interpolation(lambda_range[i], net, x1, x2, device)
            # print("interpolated latent code:", inter_latent)
            volume = predict(net, inter_latent, nb_grid, device, interp=True)
            verts, faces, normals, _ = measure.marching_cubes_lewiner(volume, 0.0, spacing=(1.0, -1.0, 1.0),
                                                                      gradient_direction='ascent')
            mesh = o3d.geometry.TriangleMesh()

            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

            o3d.io.write_triangle_mesh(
                'output' + save_fold + '/mesh_{}_{}_interp_{}_{}_{}.ply'.format(split, checkpoint, conditioned_ind1,
                                                                                conditioned_ind2, i), mesh)
