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
from torch import distributions as dist

from utils import dataset
from training_vae import build_network
from training_vae import train


def normalize_data(input_data):
    """

    Args:
        input_data: raw data, size=(batch_size, point_samples, point_dimension)

    Returns:
        output_data: normalized data between [-1, 1]

    """
    output_data = np.zeros_like(input_data)
    for i in range(len(input_data)):
        pts = input_data[i]
        size = pts.max(axis=0) - pts.min(axis=0)
        pts = 2 * pts / size.max()
        pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
        output_data[i] = pts

    return output_data


def get_prior_z(device, z_dim=128):
    p0_z = dist.Normal(torch.zeros(z_dim, device=device), torch.ones(z_dim, device=device))

    return p0_z


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 4000
    eik_weight = 0.1
    kl_weight = 1.0e-3
    use_kl = False
    use_normal = True
    z_dim = 256
    points_batch = 3000
    batch_size = 2
    save_fold = '/exp/shapenet_car_zdim_256'
    os.makedirs('models' + save_fold, exist_ok=True)

    # create prior distribution p0_z for latent code z
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim, use_kl=use_kl)

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
                                            split='train',
                                            with_normals=use_normal, points_batch=points_batch, partial_input=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

    # create optimizer
    lr = 5e-4
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print("Training!")
    avg_training_loss = []
    rec_training_loss = []
    eik_training_loss = []
    kl_training_loss = []
    for epoch in range(num_epochs):
        if epoch % 500 == 0 and epoch >= 1000:
            optimizer.defaults['lr'] = lr / 2
        avg_loss, rec_loss, eik_loss, kl_loss = train(net, train_loader, optimizer, device, eik_weight, kl_weight,
                                                      use_normal)
        avg_training_loss.append(avg_loss)
        rec_training_loss.append(rec_loss)
        eik_training_loss.append(eik_loss)
        kl_training_loss.append(kl_loss)
        print('Epoch [%d / %d] average training loss: %f rec loss: %f eik loss: %f kl loss %f' % (
            epoch + 1, num_epochs, avg_loss, rec_loss, eik_loss, kl_loss))
        if epoch % 100 == 0 and epoch:
            torch.save(net.state_dict(), 'models' + save_fold + '/model_{0:04d}.pth'.format(epoch))

    torch.save(net.state_dict(), 'models' + save_fold + '/model_final.pth')

    import matplotlib.pyplot as plt

    fig = plt.figure()
    p1, = plt.plot(avg_training_loss)
    p2, = plt.plot(rec_training_loss)
    p3, = plt.plot(eik_training_loss)
    p4, = plt.plot(kl_training_loss)
    plt.legend([p1, p2, p3, p4], ["total_loss", "rec_loss", "eik_loss", "kl_loss"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('models' + save_fold + '/loss.png')
