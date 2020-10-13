import torch
import numpy as np
from scipy import spatial


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pts, normal, knn, samples=8 ** 2, transform=None):

        B, S, D = pts.shape
        self.pts = np.zeros_like(pts)
        self.normal = normal
        self.radius = np.zeros((B, S, 1))
        self.samples = samples

        for batch in range(B):
            tree = spatial.cKDTree(pts[batch])
            dists, _ = tree.query(pts[batch], k=knn)
            radius = dists[:, -1]

            self.transform = transform
            self.pts[batch] = pts[batch].astype(np.float32)
            self.radius[batch] = np.expand_dims(radius, axis=1).astype(np.float32)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        p = self.pts[idx]
        r = self.radius[idx]
        n = self.normal[idx]
        random_idx = (torch.rand(self.samples) * p.shape[0]).long()
        p = torch.index_select((torch.from_numpy(p)).float(), 0, random_idx)
        r = torch.index_select((torch.from_numpy(r)).float(), 0, random_idx)
        n = torch.index_select((torch.from_numpy(n)).float(), 0, random_idx)

        if self.transform:
            p = self.transform(p)

        return p, r, n
