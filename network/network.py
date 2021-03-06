import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import network.latent_encoder as le
import numpy as np


def input_encoder(x, a, b):
    """
    Fourier feature mappings
    from https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_shape_occupancy.ipynb

    """
    x_proj = (2. * np.pi * x) @ b.T
    return torch.cat([a * torch.sin(x_proj), a * torch.cos(x_proj)], dim=-1) / a.norm()


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    """
    sine activation and initialization scheme
    from https://github.com/vsitzmann/siren/blob/master/modules.py

    """

    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class Decoder(nn.Module):
    def __init__(self, *args, input_dim, z_dim=128, beta=None, skip_connection=True):
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

        if str(beta).isdigit():
            self.activation = nn.Softplus(beta=beta)
        elif beta == 'sine':
            self.activation = nn.ReLU(inplace=True)
            self.activation_last = Sine()
            # self.l1.apply(first_layer_sine_init)
            # self.l2.apply(sine_init)
            # self.l3.apply(sine_init)
            # self.l4.apply(sine_init)
            # self.l5.apply(sine_init)
            # self.l6.apply(sine_init)
            # self.l7.apply(sine_init)
            # self.l_out.apply(sine_init)
        else:
            self.activation = nn.ReLU(inplace=True)
            self.l1.apply(init_weights_normal)
            self.l2.apply(init_weights_normal)
            self.l3.apply(init_weights_normal)
            self.l4.apply(init_weights_normal)
            self.l5.apply(init_weights_normal)
            self.l6.apply(init_weights_normal)
            self.l7.apply(init_weights_normal)
            self.l_out.apply(init_weights_normal)

    def forward(self, x, z):
        if self.input_mapping:
            x = input_encoder(x, self.avals.to(x.device), self.bvals.to(x.device))
        x = torch.cat((x, z), dim=-1)

        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        if self.skip_connection:
            h = torch.cat((h, x), dim=-1)
        h = self.activation(self.l5(h))
        h = self.activation(self.l6(h))
        if hasattr(self, 'activation_last'):
            h = self.activation_last(self.l7(h))
        else:
            h = self.activation(self.l7(h))
        h = self.l_out(h)

        return h


class Network(nn.Module):
    """
    Args:
        p0_z (dist): prior distribution for latent code z
    """

    def __init__(self, *args, input_dim, p0_z=None, z_dim=128, beta=None, skip_connection=True, variational=False,
                 use_kl=False):
        super(Network, self).__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))
        self.p0_z = p0_z
        self.use_kl = use_kl
        self.z_dim = z_dim
        self.vae = variational

        self.encoder = le.Encoder(dim=input_dim, z_dim=z_dim)
        self.decoder = Decoder(*args, input_dim=input_dim, z_dim=z_dim, beta=beta, skip_connection=skip_connection)

    def forward(self, *mnfld_pnts_reflected, mnfld_pnts, non_mnfld_pnts):

        if self.vae:
            mean_z, logstd_z = self.encoder(mnfld_pnts)
            q_z = dist.Normal(mean_z, torch.exp(logstd_z))
            z = q_z.rsample()
            if self.use_kl == 'kl':
                device = q_z.mean.device
                p0_z = dist.Normal(torch.zeros(self.z_dim, device=device), torch.ones(self.z_dim, device=device))
                vae_loss = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
            elif self.use_kl == 'both':
                alpha = 0.1
                mean_l2 = mean_z.norm(2, dim=-1) ** 2
                mean_l1 = mean_z.norm(1, dim=-1)
                logstd_l2 = (logstd_z + 1).norm(2, dim=-1) ** 2
                logstd_l1 = (logstd_z + 1).norm(1, dim=-1)
                vae_loss = alpha * (mean_l1 + logstd_l1) + (1 - alpha) * (mean_l2 + logstd_l2)
            else:
                vae_loss = mean_z.abs().mean(dim=-1) + (logstd_z + 1).abs().mean(dim=-1)
        else:
            z, _ = self.encoder(mnfld_pnts)
            vae_loss = torch.zeros([], device=z.device)

        z_mnfld = z.unsqueeze(dim=1).expand((-1, mnfld_pnts.shape[1], -1))
        z_non_mnfld = z.unsqueeze(dim=1).expand((-1, non_mnfld_pnts.shape[1], -1))

        h_mnfld = self.decoder(mnfld_pnts, z_mnfld)
        h_non_mnfld = self.decoder(non_mnfld_pnts, z_non_mnfld)
        if mnfld_pnts_reflected:
            h_mnfld_reflected = self.decoder(mnfld_pnts_reflected[0], z_mnfld)
        else:
            h_mnfld_reflected = None

        return h_mnfld, h_mnfld_reflected, h_non_mnfld, vae_loss

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
