import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import network.latent_encoder as le
import numpy as np


def input_encoder(x, a, b):
    x_proj = (2. * np.pi * x) @ b.T
    return torch.cat([a * torch.sin(x_proj), a * torch.cos(x_proj)], dim=-1) / a.norm()


class Decoder(nn.Module):
    def __init__(self, *args, input_dim, z_dim=128, skip_connection=True):
        super(Decoder, self).__init__()
        self.skip_connection = skip_connection
        try:
            self.input_mapping = args[0]
        except IndexError:
            self.input_mapping = False
        if self.input_mapping:
            self.avals = args[1]
            self.bvals = args[2]
            self.mapped_input_dim = args[3]

        if self.input_mapping:
            self.l1 = nn.Linear(self.mapped_input_dim + z_dim, 512)
        else:
            self.l1 = nn.Linear(input_dim + z_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        if self.skip_connection:
            if self.input_mapping:
                self.l4 = nn.Linear(512, 512 * 2 - self.mapped_input_dim - z_dim)
            else:
                self.l4 = nn.Linear(512, 512 - input_dim - z_dim)
        else:
            self.l4 = nn.Linear(512, 512)
        if self.skip_connection and self.input_mapping:
            self.l5 = nn.Linear(512 * 2, 512)
        else:
            self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l_out = nn.Linear(512, 1)

    def forward(self, x, z):
        if self.input_mapping:
            x = input_encoder(x, self.avals.to(x.device), self.bvals.to(x.device))
        x = torch.cat((x, z), dim=-1)

        h = F.softplus(self.l1(x), beta=100)
        h = F.softplus(self.l2(h), beta=100)
        h = F.softplus(self.l3(h), beta=100)
        h = F.softplus(self.l4(h), beta=100)
        if self.skip_connection:
            h = torch.cat((h, x), dim=-1)
        h = F.softplus(self.l5(h), beta=100)
        h = F.softplus(self.l6(h), beta=100)
        h = F.softplus(self.l7(h), beta=100)
        h = self.l_out(h)

        return h


class Network(nn.Module):
    """
    Args:
        p0_z (dist): prior distribution for latent code z
    """

    def __init__(self, *args, input_dim, p0_z=None, z_dim=128, skip_connection=True, variational=False, use_kl=False):
        super(Network, self).__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))
        self.p0_z = p0_z
        self.use_kl = use_kl
        self.z_dim = z_dim
        self.vae = variational

        self.encoder = le.Encoder(dim=input_dim, z_dim=z_dim)
        self.decoder = Decoder(*args, input_dim=input_dim, z_dim=z_dim, skip_connection=skip_connection)

    def forward(self, mnfld_pnts, non_mnfld_pnts):

        vae_loss = torch.zeros(1)
        if self.vae:
            mean_z, logstd_z = self.encoder(mnfld_pnts)
            q_z = dist.Normal(mean_z, torch.exp(logstd_z))
            z = q_z.rsample()
            if self.use_kl:
                device = q_z.mean.device
                p0_z = dist.Normal(torch.zeros(self.z_dim, device=device), torch.ones(self.z_dim, device=device))
                vae_loss = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            else:
                vae_loss = mean_z.abs().mean(dim=-1) + (logstd_z + 1).abs().mean(dim=-1)
        else:
            z, _ = self.encoder(mnfld_pnts)

        z_mnfld = z.unsqueeze(dim=1).expand((-1, mnfld_pnts.shape[1], -1))
        z_non_mnfld = z.unsqueeze(dim=1).expand((-1, non_mnfld_pnts.shape[1], -1))

        h_mnfld = self.decoder(mnfld_pnts, z_mnfld)
        h_non_mnfld = self.decoder(non_mnfld_pnts, z_non_mnfld)

        return h_mnfld, h_non_mnfld, vae_loss

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        """ Returns z from prior distribution.

        Args:
            size (Size): size of z
            sample (bool): whether to sample
        """
        if sample:
            z = self.p0_z.sample(size)
        else:
            z = self.p0_z.mean
            z = z.expand(*size, *z.size())

        return z
