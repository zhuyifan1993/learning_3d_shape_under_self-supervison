import os
import numpy as np

import torch
import torch.optim as optim
from torch import distributions as dist

from dataset import Dataset
from training_vae import build_network
from training_vae import train


def generate_data(nb_data=128, noise=0.0):
    t = 2 * np.random.rand(nb_data) * np.pi
    r = 1.0 + np.random.randn(nb_data) * noise
    x_p = np.stack((r * np.cos(t), r * np.sin(t)), axis=1)

    # scale the circle randomly with factor in [0.5,1)
    x_p = (np.random.rand() + 1) / 2 * x_p

    # get sparse samples
    x_p = x_p[::5]

    return x_p


def generate_data_square(nb_data=128, std=0.0):
    # x = 2 * np.random.rand(nb_data) - 1
    # y = 1 + np.random.randn(nb_data) * std
    # x_p = np.stack((x, y), axis=1)
    # x = 1 + np.random.randn(nb_data) * std
    # y = 2 * np.random.rand(nb_data) - 1
    # x_p = np.vstack([x_p, np.stack((x, y), axis=1)])

    # create partial sampling by cutting a quarter of square
    x = -np.random.rand(int(nb_data / 2))  # [-1, 0]
    y = 1 + np.random.randn(int(nb_data / 2)) * std
    x_p = np.stack((x, y), axis=1)
    x = 1 + np.random.randn(int(nb_data / 2)) * std
    y = -np.random.rand(int(nb_data / 2))  # [-1, 0]
    x_p = np.vstack([x_p, np.stack((x, y), axis=1)])

    x = -1 + np.random.randn(nb_data) * std
    y = 2 * np.random.rand(nb_data) - 1
    x_p = np.vstack([x_p, np.stack((x, y), axis=1)])
    x = 2 * np.random.rand(nb_data) - 1
    y = -1 + np.random.randn(nb_data) * std
    x_p = np.vstack([x_p, np.stack((x, y), axis=1)])

    # scale the square randomly with factor in [0.5,1)
    x_p = (np.random.rand() + 1) / 2 * x_p

    # rotation the square randomly with theta in [0,pi)
    theta = np.random.rand() * np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    for i in range(len(x_p)):
        x_p[i] = R @ x_p[i]

    # get sparse samples
    x_p = x_p[::5]

    return x_p


def get_prior_z(device, z_dim=128):
    p0_z = dist.Normal(torch.zeros(z_dim, device=device), torch.ones(z_dim, device=device))

    return p0_z


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    z_dim = 8
    save_fold = '/vae/object_200_zdim_8_partial'

    # generate synthetic data
    obj_list = []
    for i in range(200):
        x = generate_data_square(nb_data=25, std=0.01)
        obj_list.append(x)
        os.makedirs('models' + save_fold + '/object', exist_ok=True)
        np.save('models' + save_fold + '/object' + '/square_{}.npy'.format(i), x)
        # x = generate_data(nb_data=100, noise=0.01)
        # obj_list.append(x)
        # np.save('models' + save_fold + '/object' + '/circle_{}.npy'.format(i), x)

    data = np.stack(obj_list, axis=0)
    print("object num:", data.shape[0], "sample num", data.shape[1])
    dataset = Dataset(data, knn=10)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # create prior distribution p0_z for latent code z
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=2, p0_z=p0_z, z_dim=z_dim)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    num_epochs = 1000
    avg_training_loss = []
    for epoch in range(num_epochs):
        avg_loss, rec_loss = train(net, data_loader, optimizer, device)
        avg_training_loss.append(avg_loss)
        print('Epoch [%d / %d] average training loss: %f average reconstruction loss: %f' % (
            epoch + 1, num_epochs, avg_training_loss[-1], rec_loss))
        # if epoch % 100 == 0:
        #     torch.save(net.state_dict(), 'models' + save_fold + '/model_{0:04d}.pth'.format(epoch))

    torch.save(net.state_dict(), 'models' + save_fold + '/model_final.pth')

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(avg_training_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('models' + save_fold + '/loss.png')
