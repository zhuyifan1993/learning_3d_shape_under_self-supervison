import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import distributions as dist
from scipy import spatial

from network_vae import Network


def sample_fake(pts, local_sigma=0.01):
    sampled = pts + torch.randn_like(pts) * local_sigma
    return sampled


def build_network(input_dim=3, p0_z=None, z_dim=128):
    net = Network(input_dim=input_dim, p0_z=p0_z, z_dim=z_dim)
    for k, v in net.named_parameters():
        if 'encoder' in k:
            pass
        else:
            if 'weight' in k:
                std = np.sqrt(2) / np.sqrt(v.shape[0])
                nn.init.normal_(v, 0.0, std)
            if 'bias' in k:
                nn.init.constant_(v, 0)
            if k == 'l_out.weight':
                std = np.sqrt(np.pi) / np.sqrt(v.shape[1])
                nn.init.constant_(v, std)
            if k == 'l_out.bias':
                nn.init.constant_(v, -1)
    return net


def train(net, data_loader, optimizer, device, eik_weight, kl_weight, use_kl, use_normal):
    net.train()

    avg_loss = 0
    rec_loss = 0
    eik_loss = 0
    kl_loss = 0
    it = 0

    for batch in data_loader:

        pts = batch['points']
        B, S, D = pts.size()
        if use_normal:
            normal = batch['normals']
        else:
            normal = torch.zeros_like(pts)

        # rad = torch.zeros([B, S, 1])
        #
        # for b in range(B):
        #     tree = spatial.cKDTree(pts[b])
        #     dists, _ = tree.query(pts[b], k=50)
        #
        #     rad[b] = torch.from_numpy(np.expand_dims(dists[:, -1], axis=1))

        # create samples from distribution D
        fake = sample_fake(pts)
        uniform = 3 * torch.rand(B, S // 8, D) - 1.5
        xv = torch.cat((fake, uniform), axis=1)

        # KL-divergence
        pts = pts.to(device)
        q_z, q_latent_mean, q_latent_std = net.infer_z(pts)
        q_latent_mean = q_latent_mean.to(device)
        q_latent_std = q_latent_std.to(device)
        z = q_z.rsample()

        if use_kl:
            kl = dist.kl_divergence(q_z, net.p0_z).sum(dim=-1)
        else:
            kl = q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1)
        kl = kl.mean()

        # reconstruction err
        z_latent = z.unsqueeze(dim=1).expand((B, S, -1)).detach().to(device)
        pts.requires_grad_()
        y = net.fcn(pts, z_latent)
        if use_normal:
            normal = normal.to(device)
            gn = autograd.grad(outputs=y, inputs=pts,
                               grad_outputs=torch.ones(y.size()).to(device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
            loss_pts = (y.abs()).mean() + (gn - normal).norm(2, dim=2).mean()
        else:
            loss_pts = (y.abs()).mean()

        # eikonal term
        z_xv = z.unsqueeze(dim=1).expand(-1, xv.shape[1], -1).detach().to(device)
        xv = xv.requires_grad_().to(device)
        f = net.fcn(xv, z_xv)
        g = autograd.grad(outputs=f, inputs=xv,
                          grad_outputs=torch.ones(f.size()).to(device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        eikonal_term = ((g.norm(2, dim=2) - 1) ** 2).mean()

        loss = loss_pts + eik_weight * eikonal_term + kl_weight * kl

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
