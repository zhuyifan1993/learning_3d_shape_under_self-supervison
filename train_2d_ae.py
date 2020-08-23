import os
import numpy as np

import torch
import torch.optim as optim

from dataset import Dataset
from training_ae import build_network
from training_ae import train


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
    x = 2 * np.random.rand(nb_data) - 1
    y = 1 + np.random.randn(nb_data) * std
    x_p = np.stack((x, y), axis=1)
    x = 2 * np.random.rand(nb_data) - 1
    y = -1 + np.random.randn(nb_data) * std
    x_p = np.vstack([x_p, np.stack((x, y), axis=1)])
    x = -1 + np.random.randn(nb_data) * std
    y = 2 * np.random.rand(nb_data) - 1
    x_p = np.vstack([x_p, np.stack((x, y), axis=1)])
    x = 1 + np.random.randn(nb_data) * std
    y = 2 * np.random.rand(nb_data) - 1
    x_p = np.vstack([x_p, np.stack((x, y), axis=1)])

    # scale the square randomly with factor in [0.5,1)
    x_p = (np.random.rand() + 1) / 2 * x_p

    # create partial sampling
    # pt = x_p[0]
    # for i in np.arange(1, len(x_p)):
    #     xx, yy = x_p[i]
    #     if xx > 0 and yy > 0:
    #         pass
    #     else:
    #         pt = np.vstack([pt, x_p[i]])
    # x_p = np.delete(pt, 0, axis=0)

    # rotation the square randomly with theta in [0,pi)
    theta = np.random.rand() * np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    for i in range(len(x_p)):
        x_p[i] = R @ x_p[i]

    # get sparse samples
    x_p = x_p[::5]

    return x_p


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    save_fold = '/ae/object_10_cdim_2'

    # generate synthetic data
    obj_list = []
    for i in range(10):
        x = generate_data_square(nb_data=25, std=0.01)
        obj_list.append(x)
        os.makedirs('models' + save_fold + '/object', exist_ok=True)
        np.save('models' + save_fold + '/object' + '/square_{}.npy'.format(i), x)
        x = generate_data(nb_data=100, noise=0.01)
        obj_list.append(x)
        np.save('models' + save_fold + '/object' + '/circle_{}.npy'.format(i), x)

    c_dim = 2
    net = build_network(input_dim=2, c_dim=c_dim)
    net.to(device)

    optimizer = optim.Adam(net.parameters())

    # create training dataset
    for obj_ind, x in enumerate(obj_list):
        dataset = Dataset(x, knn=10)
        if obj_ind == 0:
            pts = torch.from_numpy(dataset.pts.astype(np.float32)).unsqueeze(dim=0)
            rad = torch.from_numpy(dataset.radius.astype(np.float32)).unsqueeze(dim=0)
        else:
            pts = torch.cat((pts, torch.from_numpy(dataset.pts.astype(np.float32)).unsqueeze(dim=0)))
            rad = torch.cat((rad, torch.from_numpy(dataset.radius.astype(np.float32)).unsqueeze(dim=0)))

    for epoch in range(400):
        rec_err, eiko_err = train(net, optimizer, device, pts, rad)
        print('epoch', epoch, 'rec_err:', rec_err, 'eiko_err:', eiko_err)
        if epoch % 100 == 0:
            torch.save(net.state_dict(), 'models' + save_fold + '/model_{0:04d}.pth'.format(epoch))

    torch.save(net.state_dict(), 'models' + save_fold + '/model_final.pth')
