#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/11/22 10:30
# file: kitti_statistics.py
# software: PyCharm
from utils import dataset
import torch
import statistics as stat
import matplotlib.pyplot as plt
import os

category = 'car'

train_dataset = dataset.KITTI360Dataset('data/KITTI-360/data_3d_pointcloud', 'train',
                                        category, evaluation=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
                                           pin_memory=True)

test_dataset = dataset.KITTI360Dataset('data/KITTI-360/data_3d_pointcloud', 'test',
                                       category, evaluation=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False,
                                          pin_memory=True)

pointsList_train = []
pointsList_test = []
for ind, data in enumerate(train_loader):
    conditioned_input = data['points_tgt']
    print("object id:", ind + 1, "sample points:", conditioned_input.shape[1])
    pointsList_train.append(conditioned_input.shape[1])
print(stat.median_low(pointsList_train))

num_bins = 50

fig, ax = plt.subplots()

for ind, data in enumerate(test_loader):
    conditioned_input = data['points_tgt']
    print("object id:", ind + 1, "sample points:", conditioned_input.shape[1])
    pointsList_test.append(conditioned_input.shape[1])

# the histogram of the data
n, bins, patches = ax.hist([pointsList_train, pointsList_test], num_bins, histtype='bar', label=['train', 'test'],
                           stacked=False)
ax.legend()

# add a 'best fit' line
ax.set_xlabel('Points of instance')
ax.set_ylabel('Number of instance')

# Tweak spacing to prevent clipping of ylabel

# fig.tight_layout()
# plt.show()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/kitti_{}_stats_{}.png'.format(category, num_bins))
