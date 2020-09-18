import os
import numpy as np

import torch
import torch.optim as optim
from torch import distributions as dist

from dataset import Dataset
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


def predict(net, conditioned_object, nb_grid, device):
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
    for p in pts:
        v = net(torch.Tensor(p).to(device), conditioned_object.to(device))
        v = v.reshape(-1).detach().cpu().numpy()
        val.append(v)
    val = np.concatenate(val)
    val = val.reshape(nb_grid, nb_grid, nb_grid)
    return val


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z_dim = 8
    save_fold = '/vae/shapenet_zdim_8_100_object_car_300point'
    # save_fold = '/ae/dpg_cdim_0_gargoyle'

    os.makedirs('models' + save_fold, exist_ok=True)

    data = np.load('shapenet/points_shapenet_32x32x32_train.npy')[0:100, ::10]
    # data = np.load('DGP/gargoyle.npy')[::2]
    # data = np.expand_dims(data, axis=0)

    print("object num:", len(data), "samples per object:", data.shape[1])
    data = normalize_data(data)
    dataset = Dataset(data, knn=50)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # create prior distribution p0_z for latent code z
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    num_epochs = 2000
    avg_training_loss = []
    for epoch in range(num_epochs):
        avg_loss, rec_loss = train(net, data_loader, optimizer, device)
        avg_training_loss.append(avg_loss)
        print('Epoch [%d / %d] average training loss: %f average reconstruction loss: %f' % (
            epoch + 1, num_epochs, avg_training_loss[-1], rec_loss))
        if epoch % 500 == 0 and epoch:
            # torch.save(net.state_dict(), 'models' + save_fold + '/model_{0:04d}.pth'.format(epoch))
            nb_grid = 128
            conditioned_object = torch.from_numpy(data[0].astype(np.float32)).unsqueeze(0)
            volume = predict(net, conditioned_object, nb_grid, device)
            np.save('models' + save_fold + '/3d_sdf_{0:04d}.npy'.format(epoch), volume)

    torch.save(net.state_dict(), 'models' + save_fold + '/model_final.pth')

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.ylim([0, 50])
    plt.plot(avg_training_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('models' + save_fold + '/loss.png')

    nb_grid = 128
    conditioned_object = torch.from_numpy(data[0].astype(np.float32)).unsqueeze(0)
    volume = predict(net, conditioned_object, nb_grid, device)
    np.save('models' + save_fold + '/3d_sdf.npy', volume)