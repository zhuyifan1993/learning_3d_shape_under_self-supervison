import torch
import torch.nn as nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class Encoder(nn.Module):
    """ Latent encoder class.
    from https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/onet/models/encoder_latent.py

    It encodes the input points and returns mean and standard deviation for the
    posterior Gaussian distribution.

    Args:
        z_dim (int): dimension of latent code z
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, z_dim=128, dim=3, hidden_dim=128):
        super().__init__()

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = nn.Linear(hidden_dim, z_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        B, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Reduce to  B x F
        net = self.pool(net, dim=1)

        mean = self.fc_mean(net)
        logstd = self.fc_logstd(net)

        return mean, logstd
