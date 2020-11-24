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
import matplotlib.cm as cm
from network.training import build_network, get_prior_z, input_encoder_param
from utils import dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

save_fold = '/exp_last/shapenet_all_zdim_256_p1_s50'
os.makedirs('output' + save_fold, exist_ok=True)

CONFIG_PATH = 'models' + save_fold + '/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# hyper-parameters
checkpoint = 'final'
split = cfg['generate']['split']
nb_grid = cfg['generate']['nb_grid']
save_mesh = cfg['generate']['save_mesh']
save_pointcloud = cfg['generate']['save_pointcloud']

z_dim = cfg['model']['z_dim']
skip_connection = cfg['model']['skip_connection']
input_mapping = cfg['training']['input_mapping']
embedding_method = cfg['training']['embedding_method']
beta = cfg['model']['beta']

partial_input = False
data_completeness = 1
data_sparsity = 50

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

if category is None:
    category = [cate[0] for cate in test_dataset.metadata.items()]

label = []
leng = []
for cate in category:
    label.append(test_dataset.metadata[cate]['name'])
    leng.append(len([shape for shape in test_dataset.shapes if shape['category'] == cate]))
print(label)
print(leng)
colors = cm.rainbow(np.linspace(0, 1, len(label)))
if eval_fullsque:
    net.eval()
    latent_code_List = []
    for ind, data in enumerate(test_loader):
        # if 0 <= ind <= 99 or 1500 <= ind <= 1599:
        conditioned_input = data['points']
        cat = data['category']
        print("object id:", ind + 1, "sample points:", conditioned_input.shape[1], 'category:', cat)

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
    ax.scatter(Y[:1355, 0], Y[:1355, 1], s=10, color=colors[0], label='chair')
    ax.scatter(Y[1355:1355 + 1499, 0], Y[1355:1355 + 1499, 1], s=10, color=colors[1], label='car')
    ax.scatter(Y[2854:2854 + 634, 0], Y[2854:2854 + 634, 1], s=10, color=colors[2], label='sofa')
    ax.scatter(Y[3488:3488 + 809, 0], Y[3488:3488 + 809, 1], s=10, color=colors[3], label='airplane')
    ax.scatter(Y[4297:4297 + 463, 0], Y[4297:4297 + 463, 1], s=10, color=colors[4], label='lamp')
    ax.scatter(Y[4760:4760 + 210, 0], Y[4760:4760 + 210, 1], s=10, color=colors[5], label='telephone')
    ax.scatter(Y[4970:4970 + 387, 0], Y[4970:4970 + 387, 1], s=10, color=colors[6], label='vessel')
    ax.scatter(Y[5357:5357 + 323, 0], Y[5357:5357 + 323, 1], s=10, color=colors[7], label='loudspeaker')
    ax.scatter(Y[5680:5680 + 314, 0], Y[5680:5680 + 314, 1], s=10, color=colors[8], label='cabinet')
    ax.scatter(Y[5994:5994 + 1701, 0], Y[5994:5994 + 1701, 1], s=10, color=colors[9], label='table')
    ax.scatter(Y[7695:7695 + 219, 0], Y[7695:7695 + 219, 1], s=10, color=colors[10], label='display')
    ax.scatter(Y[7914:7914 + 363, 0], Y[7914:7914 + 363, 1], s=10, color=colors[11], label='bench')
    ax.scatter(Y[8277:, 0], Y[8277:, 1], s=10, color=colors[12], label='rifle')

    # ax.set_box_aspect(3 / 1)
    # ax.set_title('t-SNE embedding visualization')
    ax.legend(loc="upper left")

    plt.savefig('output' + save_fold + '/tSNE_{}_{}.png'.format(data_completeness, data_sparsity))
