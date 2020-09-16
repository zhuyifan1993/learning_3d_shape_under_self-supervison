import numpy as np
import torch
from training_vae import build_network
from train_2d_vae import get_prior_z, generate_data_square, generate_data
import matplotlib.pyplot as plt


def predict(device, net, point_cloud):
    x = np.linspace(-1.5, 1.5, 40)
    y = np.linspace(-1.5, 1.5, 40)
    X, Y = np.meshgrid(x, y)

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    pts = np.stack((X, Y), axis=1)

    net.eval()
    val = net(torch.Tensor(pts).to(device), point_cloud)
    val = val.reshape(40, 40).detach().cpu().numpy()

    return val


def plot_data(x, v):
    plt.figure(figsize=(6, 6))
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])

    xx = np.linspace(-1.5, 1.5, 40)
    yy = np.linspace(-1.5, 1.5, 40)
    X, Y = np.meshgrid(xx, yy)
    plt.contourf(X, Y, v)
    C = plt.contour(X, Y, v)
    plt.clabel(C, inline=True, fontsize=12, colors='w')
    plt.scatter(x[:, 0], x[:, 1], color='r')
    plt.show()


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    save_fold = '/vae/object_200_zdim_8_partial_4000ep'

    zdim = 8
    p0_z = get_prior_z(device, z_dim=zdim)
    net = build_network(input_dim=2, p0_z=p0_z, z_dim=zdim)
    net.load_state_dict(torch.load('./models' + save_fold + '/model_final.pth', map_location='cpu'))

    # predict learned SDF conditioned on input pointcloud
    x = generate_data_square(nb_data=25, std=0.01)
    # x = generate_data(nb_data=100, noise=0.01)
    # x = np.load('models' + save_fold + '/object' + '/square_88.npy')
    point_cloud = torch.from_numpy(x.astype(np.float32)).unsqueeze(dim=0)
    v = predict(device, net, point_cloud)

    plot_data(x, v)
