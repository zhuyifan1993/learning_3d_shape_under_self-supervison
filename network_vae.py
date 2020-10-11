import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import latent_encoder as le


class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim=128):
        super(Decoder, self).__init__()

        self.l1 = nn.Linear(input_dim + z_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512 - input_dim - z_dim)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l_out = nn.Linear(512, 1)

    def forward(self, x, z):
        x = torch.cat((x, z), axis=2)

        h = F.softplus(self.l1(x), beta=100)
        h = F.softplus(self.l2(h), beta=100)
        h = F.softplus(self.l3(h), beta=100)
        h = F.softplus(self.l4(h), beta=100)
        h = torch.cat((h, x), axis=2)
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

    def __init__(self, input_dim, p0_z=None, z_dim=128, use_kl=False):
        super(Network, self).__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))
        self.p0_z = p0_z
        self.use_kl = use_kl
        self.z_dim = z_dim

        self.encoder = le.Encoder(dim=input_dim, z_dim=z_dim)
        self.decoder = Decoder(input_dim=input_dim, z_dim=z_dim)

    def forward(self, mnfld_pnts, non_mnfld_pnts):

        mean_z, logstd_z = self.encoder(mnfld_pnts)
        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        z = q_z.rsample()
        if self.use_kl:
            device = q_z.mean.device
            p0_z = dist.Normal(torch.zeros(self.z_dim, device=device), torch.ones(self.z_dim, device=device))
            vae_loss = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
        else:
            vae_loss = mean_z.abs().mean(dim=-1) + (logstd_z + 1).abs().mean(dim=-1)
        if self.z_dim == 0:
            vae_loss = torch.zeros(1)

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

# former network
# class Network(nn.Module):
#     """
#     Args:
#         p0_z (dist): prior distribution for latent code z
#     """
#
#     def __init__(self, input_dim, p0_z=None, z_dim=128, use_kl=None):
#         super(Network, self).__init__()
#         if p0_z is None:
#             p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))
#         self.p0_z = p0_z
#
#         self.encoder = le.Encoder(dim=input_dim, z_dim=z_dim)
#         self.l1 = nn.Linear(input_dim + z_dim, 512)
#         self.l2 = nn.Linear(512, 512)
#         self.l3 = nn.Linear(512, 512)
#         self.l4 = nn.Linear(512, 512 - input_dim - z_dim)
#         self.l5 = nn.Linear(512, 512)
#         self.l6 = nn.Linear(512, 512)
#         self.l7 = nn.Linear(512, 512)
#         self.l_out = nn.Linear(512, 1)
#
#     def forward(self, x, point_cloud):
#         pointsize = x.size(0)
#         x = x.unsqueeze(dim=0)
#
#         # sample z from prior p(z)
#         # z = self.get_z_from_prior()
#         # z = point_cloud
#         # z = z.unsqueeze(0).unsqueeze(0).expand((1, pointsize, -1))
#
#         # sample z from posterior q(z|x)
#         # q_z = self.infer_z(point_cloud)
#         # z = q_z.sample().unsqueeze(dim=1).expand((1, pointsize, -1))
#         mean_z, _ = self.encoder(point_cloud)
#         # print(mean_z)
#         z = mean_z.unsqueeze(dim=1).expand((1, pointsize, -1))
#
#         h = self.fcn(x, z).squeeze(dim=0)
#         return h
#
#     def infer_z(self, point_cloud):
#         mean_z, logstd_z = self.encoder(point_cloud)
#         q_z = dist.Normal(mean_z, torch.exp(logstd_z))
#         return q_z, mean_z, logstd_z
#
#     def fcn(self, x, z):
#         x = torch.cat((x, z), axis=2)
#
#         h = F.softplus(self.l1(x), beta=100)
#         h = F.softplus(self.l2(h), beta=100)
#         h = F.softplus(self.l3(h), beta=100)
#         h = F.softplus(self.l4(h), beta=100)
#         h = torch.cat((h, x), axis=2)
#         h = F.softplus(self.l5(h), beta=100)
#         h = F.softplus(self.l6(h), beta=100)
#         h = F.softplus(self.l7(h), beta=100)
#         h = self.l_out(h)
#
#         return h
