import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import distributions as dist

from network_vae import Network


def sample_fake(pts, noise=0.3):
    sampled = pts + torch.normal(0, 1, pts.shape) * noise.unsqueeze(dim=2)
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


def train(net, optimizer, device, pts, rad):
    net.train()

    batchsize = pts.size(0)
    pointsize = pts.size(1)

    optimizer.zero_grad()

    # create samples from distribution D
    fake = torch.Tensor(sample_fake(pts, rad).float())
    uniform = 3 * torch.rand_like(fake) - 1.5
    fake = torch.cat((fake, uniform), axis=1)
    xv = fake.requires_grad_(True)
    xv = xv.to(device)

    # KL-divergence
    pts = pts.to(device)
    q_z = net.infer_z(pts)
    z = q_z.rsample().unsqueeze(dim=1).expand((batchsize, pointsize, -1))
    kl = dist.kl_divergence(q_z, net.p0_z).sum(dim=-1)
    kl = kl.mean()

    # reconstruction err
    z = z.to(device)
    y = net.fcn(pts, z)
    loss_pts = (y ** 2).mean()

    # eikonal term
    z_xv = q_z.sample().unsqueeze(dim=1).expand((batchsize, pointsize * 2, -1))
    z_xv = z_xv.to(device)
    f = net.fcn(xv, z_xv)
    g = autograd.grad(outputs=f, inputs=xv,
                      grad_outputs=torch.ones(f.size()).to(device),
                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    eikonal_term = ((g.norm(2, dim=2) - 1) ** 2).mean()

    loss = loss_pts + eikonal_term + kl * 0.01
    loss.backward()
    optimizer.step()

    rec_err = loss_pts.item()
    eiko_err = eikonal_term.item()
    kl_div = kl.item()

    return rec_err, eiko_err, kl_div
