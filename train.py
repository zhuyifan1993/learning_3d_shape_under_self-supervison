#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/10/2 2:56
# file: train.py
# software: PyCharm
import os
import glob
import yaml
import numpy as np

import torch
import torch.optim as optim

from utils import dataset
from network.training import get_prior_z, build_network, train, input_encoder_param

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    CONFIG_PATH = 'configs/shapenet.yaml'
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # hyper-parameters
    num_epochs = cfg['training']['epochs']
    points_batch = cfg['training']['subsamples_each_step']
    batch_size = cfg['training']['batch_size']
    lr = cfg['training']['lr']
    use_lr_schedule = cfg['training']['lr_schedule']
    retrieve_model = cfg['training']['retrieve_model']

    use_eik = cfg['model']['use_eik']
    variational = cfg['model']['variational']
    use_kl = cfg['model']['use_kl']
    eik_weight = cfg['model']['eik_weight']
    vae_weight = cfg['model']['vae_weight']
    z_dim = cfg['model']['z_dim']

    geo_initial = cfg['training']['geo_initial']
    use_normal = cfg['training']['use_normal']
    enforce_symmetry = cfg['training']['enforce_symmetry']

    skip_connection = cfg['model']['skip_connection']
    input_mapping = cfg['training']['input_mapping']
    embedding_method = cfg['training']['embedding_method']
    beta = cfg['model']['beta']

    partial_input = cfg['data']['partial_input']
    data_completeness = cfg['data']['data_completeness']
    data_sparsity = cfg['data']['data_sparsity']

    # save folder
    save_fold = cfg['dir']['save_fold']
    os.makedirs('models' + save_fold, exist_ok=True)

    # save config file
    f = open('models' + save_fold + '/config.yaml', "w")
    yaml.dump(cfg, f)
    f.close()

    # input mapping
    args = ()
    if input_mapping:
        args = input_encoder_param(input_mapping, embedding_method, device)

    # build network
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(*args, input_dim=3, p0_z=p0_z, z_dim=z_dim, beta=beta, skip_connection=skip_connection,
                        variational=variational, use_kl=use_kl, geo_initial=geo_initial)

    if retrieve_model:
        model_path = cfg['training']['retrieve_path']
        checkpoint = cfg['training']['checkpoint']
        saved_model_state = torch.load('models' + model_path + '/model_{}.pth'.format(checkpoint), map_location='cpu')
        net.load_state_dict({k.replace('module.', ''): v for k, v in saved_model_state.items()})

    # set multi-gpu if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    # create dataloader
    DATA_PATH = cfg['data']['path']
    fields = {
        'inputs': dataset.PointCloudField(cfg['data']['pointcloud_file'])
    }
    category = cfg['data']['classes']
    train_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=category,
                                            split='train', with_normals=use_normal, points_batch=points_batch,
                                            partial_input=partial_input, data_completeness=data_completeness,
                                            data_sparsity=data_sparsity)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    print(optimizer)

    print("Training!")
    avg_training_loss = []
    rec_training_loss = []
    eik_training_loss = []
    vae_training_loss = []
    for epoch in range(num_epochs):
        if use_lr_schedule:
            if epoch % 500 == 0 and epoch >= 1000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr / (2 ** (epoch // 500 - 1))
                print(optimizer)
        avg_loss, rec_loss, eik_loss, vae_loss = train(net, train_loader, optimizer, device, eik_weight, vae_weight,
                                                       use_normal, use_eik, enforce_symmetry)
        avg_training_loss.append(avg_loss)
        rec_training_loss.append(rec_loss)
        eik_training_loss.append(eik_loss)
        vae_training_loss.append(vae_loss)
        print('Epoch [%d / %d] average training loss: %f rec loss: %f eik loss: %f vae loss %f' % (
            epoch + 1, num_epochs, avg_loss, rec_loss, eik_loss, vae_loss))
        if epoch % 100 == 0 and epoch:
            torch.save(net.state_dict(), 'models' + save_fold + '/model_{0:04d}.pth'.format(epoch))

    torch.save(net.state_dict(), 'models' + save_fold + '/model_final.pth')

    # plot loss
    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1, = plt.plot(avg_training_loss)
    p2, = plt.plot(rec_training_loss)
    p3, = plt.plot(eik_training_loss)
    p4, = plt.plot(vae_training_loss)
    plt.legend([p1, p2, p3, p4], ["total_loss", "rec_loss", "eik_loss", "vae_loss"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('models' + save_fold + '/loss.png')
