import torch
import numpy as np
from scipy import spatial


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pts, knn, transform=None):

        B, S, D = pts.shape
        self.pts = np.zeros_like(pts)
        self.radius = np.zeros((B, S, 1))

        for batch in range(B):
            tree = spatial.KDTree(pts[batch])
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

        if self.transform:
            p = self.transform(p)

        return p, r
