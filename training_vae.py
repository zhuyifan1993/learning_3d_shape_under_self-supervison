import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import distributions as dist

from network_vae import Network


def sample_fake(pts, noise=0.3):
    sampled = pts + torch.normal(0, 1, pts.shape) * noise
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
        pts = batch[0].float()
        rad = batch[1].float()
        normal = batch[2].float()
        B, S, _ = pts.size()

        # create samples from distribution D
        fake = torch.Tensor(sample_fake(pts, rad))
        uniform = 3 * torch.rand_like(fake) - 1.5
        fake = torch.cat((fake, uniform), axis=1)
        xv = fake.requires_grad_(True)
        xv = xv.to(device)

        # KL-divergence
        pts = pts.to(device)
        q_z, q_latent_mean, q_latent_std = net.infer_z(pts)
        q_latent_mean = q_latent_mean.to(device)
        q_latent_std = q_latent_std.to(device)
        z = q_z.rsample()
        z_latent = z.unsqueeze(dim=1).expand((B, S, -1))
        if use_kl:
            kl = dist.kl_divergence(q_z, net.p0_z).sum(dim=-1)
        else:
            kl = q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1)
        kl = kl.mean()

        # reconstruction err
        z_latent = z_latent.to(device)
        normal = normal.to(device)
        pts_xv = pts.requires_grad_(True)
        pts_xv = pts_xv.to(device)
        y = net.fcn(pts_xv, z_latent)
        if use_normal:
            gn = autograd.grad(outputs=y, inputs=pts_xv,
                               grad_outputs=torch.ones(y.size()).to(device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
            loss_pts = (y ** 2).mean() + (gn - normal).norm(2, dim=2).mean()
        else:
            loss_pts = (y ** 2).mean()

        # eikonal term
        z_xv = z.unsqueeze(dim=1).expand((B, S * 2, -1))
        z_xv_no_grad = z_xv.detach().to(device)
        f = net.fcn(xv, z_xv_no_grad)
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
