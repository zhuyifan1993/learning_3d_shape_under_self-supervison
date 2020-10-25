import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy import spatial
from torch import distributions as dist

from network.network import Network


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


def sample_fake(pts, local_sigma=0.01):
    """
    create samples from distribution D (gaussians and uniform dist)
    Args:
        pts:
        local_sigma:

    Returns:

    """
    B, S, D = pts.size()
    gaussians = pts + torch.randn_like(pts) * local_sigma
    uniform = 3 * torch.rand(B, S // 8, D) - 1.5
    sampled = torch.cat([gaussians, uniform.to(pts.device)], axis=1)
    return sampled


def calculate_local_sigma(pts, knn):
    B, S, D = pts.size()
    rad = torch.zeros([B, S, 1])

    for b in range(B):
        tree = spatial.cKDTree(pts[b])
        dists, _ = tree.query(pts[b], k=knn)

        rad[b] = torch.from_numpy(np.expand_dims(dists[:, -1], axis=1))

    return rad


def input_encoder(x, a, b):
    return (np.concatenate([a * np.sin((2. * np.pi * x) @ b.T),
                            a * np.cos((2. * np.pi * x) @ b.T)], axis=-1) / np.linalg.norm(
        a)) if a is not None else (x * 2. - 1.)


def build_network(input_dim=3, p0_z=None, z_dim=128, skip_connection=True, variational=False, use_kl=False,
                  geo_initial=True):
    net = Network(input_dim=input_dim, p0_z=p0_z, z_dim=z_dim, skip_connection=skip_connection, variational=variational,
                  use_kl=use_kl)
    if geo_initial:
        print("Perform geometric initialization!\n")
        for k, v in net.named_parameters():
            if 'encoder' in k:
                pass
            else:
                if 'weight' in k:
                    std = np.sqrt(2) / np.sqrt(v.shape[0])
                    nn.init.normal_(v, 0.0, std)
                if 'bias' in k:
                    nn.init.constant_(v, 0)
                if 'l_out.weight' in k:
                    std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
                    nn.init.constant_(v, std)
                if 'l_out.bias' in k:
                    nn.init.constant_(v, -0.5)
    return net


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def train(net, data_loader, optimizer, device, eik_weight, kl_weight, use_normal, input_mapping):
    net.train()

    avg_loss = 0
    rec_loss = 0
    eik_loss = 0
    kl_loss = 0
    it = 0

    if input_mapping:
        embedding_size = 256
        embedding_method = 'gauss'
        embedding_param = 12.
        embed_params = [embedding_method, embedding_size, embedding_param]
        embedding_method, embedding_size, embedding_scale = embed_params
        rs = np.random.RandomState(0)

        if embedding_method == 'gauss':
            print('gauss bvals')
            bvals = rs.normal(size=[embedding_size, 3]) * embedding_scale

        if embedding_method == 'posenc':
            print('posenc bvals')
            bvals = 2. ** np.linspace(0, embedding_scale, embedding_size // 3) - 1
            bvals = np.reshape(np.eye(3) * bvals[:, None, None], [len(bvals) * 3, 3])
        if embedding_method == 'basic':
            print('basic bvals')
            bvals = np.eye(3)

        if embedding_method == 'none':
            print('NO abvals')
            avals = None
            bvals = None
        else:
            avals = np.ones_like(bvals[:, 0])

    for batch in data_loader:

        pts = batch['points']
        if input_mapping:
            pts = input_encoder(pts.numpy(), avals, bvals).astype(np.float32)
            pts = torch.from_numpy(pts)

        # rad = calculate_local_sigma(pts, 50).to(device)

        # create samples from distribution D
        pts = pts.to(device)
        fake = sample_fake(pts)

        # forward
        pts.requires_grad_()
        fake.requires_grad_()
        h_mnfld, h_non_mnfld, kl = net(pts, fake)

        # vae loss
        kl = kl.mean()

        # reconstruction loss
        if use_normal:
            normal = batch['normals'].to(device)
            pts_grad = gradient(inputs=pts, outputs=h_mnfld)
            loss_pts = h_mnfld.abs().mean() + (pts_grad - normal).norm(2, dim=2).mean()
        else:
            loss_pts = h_mnfld.abs().mean()

        # eikonal loss
        fake_grad = gradient(inputs=fake, outputs=h_non_mnfld)
        eikonal_term = ((fake_grad.norm(2, dim=2) - 1) ** 2).mean()

        loss = loss_pts + eik_weight * eikonal_term + kl_weight * kl

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        rec_loss += loss_pts.item()
        eik_loss += eikonal_term.item()
        kl_loss += kl.item()
        it += 1

    avg_loss /= it
    rec_loss /= it
    eik_loss /= it
    kl_loss /= it

    return avg_loss, rec_loss, eik_loss, kl_loss
