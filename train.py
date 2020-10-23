#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/10/2 2:56
# file: train.py
# software: PyCharm
import os
import glob
import numpy as np

import torch
import torch.optim as optim

from utils import dataset
from network.training import get_prior_z, build_network, train

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # hyper-parameters
    num_epochs = 2000
    eik_weight = 0.1
    vae_weight = 1.0e-3
    variational = True
    use_kl = False
    use_normal = True
    partial_input = True
    data_completeness = 0.7
    data_sparsity = 1
    geo_initial = True
    z_dim = 256
    points_batch = 3000
    batch_size = 2
    lr = 5e-4

    # save folder
    save_fold = '/debug/shapenet_car_zdim_256'
    os.makedirs('models' + save_fold, exist_ok=True)

    # build network
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim, variational=variational, use_kl=use_kl,
                        geo_initial=geo_initial)

    # set multi-gpu if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    # create dataloader
    DATA_PATH = 'data/ShapeNet'
    fields = {'inputs': dataset.PointCloudField('pointcloud.npz')}
    train_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=['02958343'],
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
        if epoch % 500 == 0 and epoch >= 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (2 ** (epoch // 500 - 1))
            print(optimizer)
        avg_loss, rec_loss, eik_loss, vae_loss = train(net, train_loader, optimizer, device, eik_weight, vae_weight,
                                                       use_normal)
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
