import glob
import os

import numpy as np

import tqdm

import torch
from torch import distributions as dist
import yaml

from network.training import build_network, get_prior_z, input_encoder_param
from utils import dataset
import utils.plots as plt


def predict(net, conditioned_input, nb_grid, device):
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
    latent_code, _ = net.encoder(conditioned_input)
    # print(latent_code)
    latent_code = latent_code.unsqueeze(dim=1).expand((-1, pts.shape[1], -1))
    for p in tqdm.tqdm(pts):
        v = net.decoder(torch.Tensor(p).unsqueeze(0).to(device), latent_code)
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    val = np.concatenate(val)
    val = val.reshape(nb_grid, nb_grid, nb_grid)
    return val


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    save_fold = '/exp_last/shapenet_sofa_kitti_building_zdim_256_ep3200'
    os.makedirs('output' + save_fold, exist_ok=True)

    # CONFIG_PATH = 'models' + save_fold + '/config.yaml'
    # with open(CONFIG_PATH, 'r') as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)

    # hyper-parameters
    checkpoint = '0100'
    split = 'test'
    nb_grid = 128
    conditioned_ind = 0
    save_mesh = True
    save_pointcloud = False

    z_dim = 256
    skip_connection = True
    input_mapping = False
    embedding_method = ''
    beta = 100

    partial_input = True
    data_completeness = 0.7
    data_sparsity = 100

    try:
        volume = np.load('sdf' + save_fold + '/sdf_{}_{}_{}.npy'.format(split, checkpoint, conditioned_ind))
    except FileNotFoundError:
        volume = None

    if volume is None:
        DATA_PATH = 'data/ShapeNet'
        fields = {
            'inputs': dataset.PointCloudField('pointcloud.npz')
        }
        category = ['02958343']
        test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=category,
                                               split=split, partial_input=partial_input,
                                               data_completeness=data_completeness, data_sparsity=data_sparsity,
                                               evaluation=True)

        # conditioned_input = test_dataset.__getitem__(conditioned_ind)['points'].unsqueeze(0)
        ds_kitti = dataset.KITTI360Dataset('data/KITTI-360/data_3d_pointcloud', 'train', 'building', evaluation=True)
        conditioned_input = ds_kitti.__getitem__(conditioned_ind)['points_tgt'].unsqueeze(0)
        print("object id:", conditioned_ind + 1, "sample points:", conditioned_input.shape[1])

        # input mapping

        args = ()
        if input_mapping:
            args = input_encoder_param(input_mapping, embedding_method, device)

        p0_z = get_prior_z(device, z_dim=z_dim)
        net = build_network(*args, input_dim=3, p0_z=p0_z, z_dim=z_dim, beta=beta, skip_connection=skip_connection,
                            geo_initial=False)
        net = net.to(device)

        saved_model_state = torch.load('models' + save_fold + '/model_{}.pth'.format(checkpoint), map_location='cpu')
        net.load_state_dict({k.replace('module.', ''): v for k, v in saved_model_state.items()})

        net.eval()
        conditioned_input = conditioned_input.to(device)
        latent_code, logstd = net.encoder(conditioned_input)

        # sampling
        # q_z = dist.Normal(torch.zeros(z_dim, device=device).unsqueeze(0),
        #                   torch.ones(z_dim, device=device).unsqueeze(0) * np.exp(0))
        # latent_code = q_z.sample()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(np.asarray(latent_code.detach()).squeeze(), bins=np.arange(-0.02, 0.02, 0.001))
        # plt.show()
        print('latent code:', latent_code)

        if not partial_input:
            input_pc = conditioned_input.squeeze()
            all_latent = latent_code.repeat(input_pc.shape[0], 1)
            points = torch.cat([all_latent, input_pc], dim=-1).detach()
            is_uniform = False
        else:
            points = None
            is_uniform = True

        surface = plt.get_surface_trace(points=points, decoder=net.decoder, latent=latent_code, resolution=nb_grid,
                                        mc_value=0, is_uniform=is_uniform, verbose=True, save_ply=True, connected=True)
        if save_mesh:
            surface['mesh_export'].export(
                'output' + save_fold + '/Tempmesh_{}_{}_{}_{}_{}.off'.format(split, data_completeness, data_sparsity,
                                                                             checkpoint, conditioned_ind), 'off')
        if save_pointcloud:
            surface['mesh_export'].export(
                'output' + save_fold + '/mesh_{}_{}_{}_{}_{}.ply'.format(split, data_completeness, data_sparsity,
                                                                         checkpoint, conditioned_ind), 'ply')
        print(surface)
