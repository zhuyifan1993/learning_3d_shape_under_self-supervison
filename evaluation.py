import glob
import os

import numpy as np

import torch

from training_vae import build_network
from train_3d_vae import normalize_data, get_prior_z


def predict(net, conditioned_input, nb_grid, device, interp=False):
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
    conditioned_input = conditioned_input.to(device)

    if interp:
        latent_code = conditioned_input.unsqueeze(dim=1).expand((-1, pts.shape[1], -1))
        for p in pts:
            v = net.module.decoder(torch.Tensor(p).unsqueeze(0).to(device), latent_code)
            v = v.reshape(-1).detach().cpu().numpy()
            val.append(v)
    else:
        latent_code, _ = net.module.encoder(conditioned_input)
        latent_code = latent_code.unsqueeze(dim=1).expand((-1, pts.shape[1], -1))
        for p in pts:
            v = net.module.decoder(torch.Tensor(p).unsqueeze(0).to(device), latent_code)
            v = v.reshape(-1).detach().cpu().numpy()
            val.append(v)
    val = np.concatenate(val)
    val = val.reshape(nb_grid, nb_grid, nb_grid)

    return val


def interpolation(lambda1, model, img1, img2, device):
    model.eval()
    # latent vector of first input
    img1 = img1.to(device)
    latent_1, _ = model.module.encoder(img1)

    # latent vector of second input
    img2 = img2.to(device)
    latent_2, _ = model.module.encoder(img2)

    # interpolation of the two latent vectors
    inter_latent = (1 - lambda1) * latent_1 + lambda1 * latent_2

    return inter_latent


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    z_dim = 256
    save_fold = '/vae/week_shapenet_car_zdim_256_bs80_ps3k'
    p0_z = get_prior_z(device, z_dim=z_dim)
    net = build_network(input_dim=3, p0_z=p0_z, z_dim=z_dim)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load('./models' + save_fold + '/model_1000.pth', map_location='cpu'))

    # path = glob.glob('shapenet_pointcloud/*')
    # for i in np.arange(0, len(path)):
    #     data_i = np.load(path[i])['points']
    #     if i == 0:
    #         data = np.expand_dims(data_i, axis=0)
    #     else:
    #         data_i = np.expand_dims(data_i, axis=0)
    #         data = np.concatenate([data, data_i])

    # data = np.load('shapenet_pointcloud/0001.npz')['points']
    # data = np.expand_dims(data, axis=0)
    # print("object num:", len(data), "samples per object:", data.shape[1])
    # data = normalize_data(data)
    # conditioned_object = torch.from_numpy(data[0].astype(np.float32)).unsqueeze(0)
    #
    # nb_grid = 128
    # volume = predict(net, conditioned_object, nb_grid, device)
    # np.save('models' + save_fold + '/3d_sdf_car0000.npy', volume)

    # Interpolate in Latent Space
    # x1 = np.load('shapenet_pointcloud/0000.npz')['points']
    # x1 = normalize_data(np.expand_dims(x1, axis=0))
    # x1 = torch.from_numpy(x1[0].astype(np.float32)).unsqueeze(0)
    # x2 = np.load('shapenet_pointcloud/0001.npz')['points']
    # x2 = normalize_data(np.expand_dims(x2, axis=0))
    # x2 = torch.from_numpy(x2[0].astype(np.float32)).unsqueeze(0)

    DATA_PATH = 'data/ShapeNet'
    split_file = os.path.join(DATA_PATH, "02958343", 'test.lst')
    with open(split_file, 'r') as f:
        model = f.read().split('\n')
    x1 = np.load(os.path.join(DATA_PATH, "02958343", model[0], 'pointcloud.npz'))['points']
    x1 = normalize_data(np.expand_dims(x1, axis=0))
    x1 = torch.from_numpy(x1.astype(np.float32))
    x2 = np.load(os.path.join(DATA_PATH, "02958343", model[769], 'pointcloud.npz'))['points']
    x2 = normalize_data(np.expand_dims(x2, axis=0))
    x2 = torch.from_numpy(x2.astype(np.float32))

    lambda_range = np.linspace(0, 1, 10)
    nb_grid = 128
    for i in range(len(lambda_range)):
        inter_latent = interpolation(lambda_range[i], net, x1, x2, device)
        print("interpolated latent code:", inter_latent)
        volume = predict(net, inter_latent, nb_grid, device, interp=True)
        np.save('models' + save_fold + '/3d_sdf_car_interp_{}.npy'.format(i), volume)
