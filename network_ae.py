import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet as pn


class Network(nn.Module):

    def __init__(self, input_dim, c_dim=512):
        super(Network, self).__init__()

        self.encoder = pn.Encoder(dim=input_dim, c_dim=c_dim)

        self.l1 = nn.Linear(input_dim + c_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512 - input_dim - c_dim)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l_out = nn.Linear(512, 1)

    def forward(self, x, point_cloud):
        B, S, _ = x.size()

        c = self.encoder(point_cloud).unsqueeze(dim=1).expand((B, S, -1))
        h = self.fcn(x, c)

        return h

    def fcn(self, x, c):
        x = torch.cat((x, c), axis=2)

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
