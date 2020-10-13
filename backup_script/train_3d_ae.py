import os
import numpy as np

import torch
import torch.optim as optim

from dataset import Dataset
from training_ae import build_network
from training_ae import train


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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    c_dim = 0
    save_fold = '/ae/shapenet_cdim_0_single_object_car'
    # save_fold = '/ae/dpg_cdim_0_gargoyle'

    os.makedirs('models' + save_fold, exist_ok=True)

    data = np.load('shapenet/points_shapenet_32x32x32_train.npy')[0]
    # data = np.load('DGP/gargoyle.npy')[::2]
    data = np.expand_dims(data, axis=0)

    print("object num:", len(data), "samples per object:", data.shape[1])
    data = normalize_data(data)
    dataset = Dataset(data, knn=50)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    net = build_network(input_dim=3, c_dim=c_dim)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(2000):
        rec_err, eiko_err = train(net, data_loader, optimizer, device)
        print('epoch', epoch, 'rec_err:', rec_err, 'eiko_err:', eiko_err)
        if epoch % 100 == 0:
            torch.save(net.state_dict(), 'models' + save_fold + '/model_{0:04d}.pth'.format(epoch))

    torch.save(net.state_dict(), 'models' + save_fold + '/model_final.pth')
