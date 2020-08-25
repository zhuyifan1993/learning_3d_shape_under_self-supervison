import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn

from network_ae import Network


def sample_fake(pts, noise=0.3):
    sampled = pts + torch.normal(0, 1, pts.shape) * noise
    return sampled


def build_network(input_dim=3, c_dim=128):
    net = Network(input_dim=input_dim, c_dim=c_dim)
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


def train(net, data_loader, optimizer, device):
    net.train()

    rec_err = 0
    eiko_err = 0
    total_batch = 0

    for batch in data_loader:
        pts = batch[0].float()
        rad = batch[1].float()
        B, S, _ = pts.size()

        optimizer.zero_grad()

        # create samples from distribution D
        fake = torch.Tensor(sample_fake(pts, rad))
        uniform = 3 * torch.rand_like(fake) - 1.5
        fake = torch.cat((fake, uniform), axis=1)
        xv = fake.requires_grad_(True)
        xv = xv.to(device)

        # reconstruction err
        pts = pts.to(device)
        y = net(pts, pts)
        loss_pts = (y ** 2).mean()

        # eikonal term
        c_xv = net.encoder(pts).unsqueeze(dim=1).expand((B, S * 2, -1))
        c_xv = c_xv.detach().to(device)
        f = net.fcn(xv, c_xv)
        g = autograd.grad(outputs=f, inputs=xv,
                          grad_outputs=torch.ones(f.size()).to(device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        eikonal_term = ((g.norm(2, dim=2) - 1) ** 2).mean()

        loss = loss_pts + 0.1 * eikonal_term
        loss.backward()
        optimizer.step()

        rec_err += loss_pts.item()
        eiko_err += eikonal_term.item()
        total_batch += B

    rec_err /= total_batch
    eiko_err /= total_batch

    return rec_err, eiko_err
