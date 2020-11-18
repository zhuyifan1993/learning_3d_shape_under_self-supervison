#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/11/17 22:40
# file: t-SNE_visua.py
# software: PyCharm
import glob
import os
import logging

import numpy as np

import torch
import yaml

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from network.training import build_network, get_prior_z, input_encoder_param
from utils import dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

save_fold = '/exp_last/shapenet_car_plane_zdim_256'
os.makedirs('output' + save_fold, exist_ok=True)

CONFIG_PATH = 'models' + save_fold + '/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# hyper-parameters
checkpoint = '1700'
split = cfg['generate']['split']
nb_grid = cfg['generate']['nb_grid']
save_mesh = cfg['generate']['save_mesh']
save_pointcloud = cfg['generate']['save_pointcloud']

z_dim = cfg['model']['z_dim']
skip_connection = cfg['model']['skip_connection']
input_mapping = cfg['training']['input_mapping']
embedding_method = cfg['training']['embedding_method']
beta = cfg['model']['beta']

partial_input = cfg['generate']['partial_input']
data_completeness = cfg['generate']['data_completeness']
data_sparsity = cfg['generate']['data_sparsity']

conditioned_ind1 = 0
conditioned_ind2 = 769
eval_fullsque = True
latentsp_interp = False

# build network
args = ()
if input_mapping:
    args = input_encoder_param(input_mapping, embedding_method, device)

p0_z = get_prior_z(device, z_dim=z_dim)
net = build_network(*args, input_dim=3, p0_z=p0_z, z_dim=z_dim, beta=beta, skip_connection=skip_connection,
                    geo_initial=False)
net = net.to(device)
saved_model_state = torch.load('models' + save_fold + '/model_{}.pth'.format(checkpoint), map_location='cpu')
net.load_state_dict({k.replace('module.', ''): v for k, v in saved_model_state.items()})

# create dataloader
DATA_PATH = cfg['data']['path']
fields = {
    'inputs': dataset.PointCloudField(cfg['data']['pointcloud_file'])
}
category = cfg['data']['classes']
test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=category,
                                       split=split, partial_input=partial_input,
                                       data_completeness=data_completeness, data_sparsity=data_sparsity,
                                       evaluation=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
                                          pin_memory=True)

if eval_fullsque:
    net.eval()
    latent_code_List = []
    for ind, data in enumerate(test_loader):
        # if 0 <= ind <= 99 or 1500 <= ind <= 1599:
        conditioned_input = data['points']
        print("object id:", ind + 1, "sample points:", conditioned_input.shape[1])

        conditioned_input = conditioned_input.to(device)
        latent_code, _ = net.encoder(conditioned_input)
        latent_code = latent_code.detach().cpu().numpy()
        latent_code_List.append(latent_code)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    latent_code = np.concatenate(latent_code_List)
    # latent_code = latent_code.detach().cpu().numpy()
    print("orginal shape:", latent_code.shape)
    Y = tsne.fit_transform(latent_code)
    print("embedded shape", Y.shape)

    fig, ax = plt.subplots()
    ax.scatter(Y[:1499, 0], Y[:1499, 1], color='r', label='car')
    ax.scatter(Y[1499:, 0], Y[1499:, 1], color='b', label='plane')

    ax.set_title('t-SNE embedding visualization')
    ax.legend()

    plt.savefig('output' + save_fold + '/tSNE.png')
