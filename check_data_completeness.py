#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/10/22 21:21
# file: check_data_completeness.py
# software: PyCharm

from utils import dataset
import torch
from tqdm import tqdm
import pandas as pd

split = 'test'
partial_input = True
data_completeness = 0.5

# create dataloader
DATA_PATH = 'data/ShapeNet'
fields = {
    'inputs': dataset.PointCloudField('pointcloud.npz')
}
test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=['02958343'],
                                       split=split, partial_input=partial_input, data_completeness=data_completeness,
                                       evaluation=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

eval_dicts = []

for it, data in enumerate(tqdm(test_loader)):
    if it == 2: break
    try:
        model_dict = test_dataset.get_model_dict(it)
    except AttributeError:
        model_dict = {'shape': str(it), 'category': 'n/a'}

    shapename = model_dict['shape']
    category_id = model_dict['category']

    try:
        category_name = test_dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    # Evaluating mesh and pointcloud
    # Start row and put basic information inside
    eval_dict = {
        'idx': it,
        'class id': category_id,
        'class name': category_name,
        'shape name': shapename,
    }
    eval_dicts.append(eval_dict)

    pointcloud_tgt = data['points_tgt'].squeeze(0).numpy()
    pointcloud_input = data['points'].squeeze(0).numpy()
    eval_dict['input/ground truth'] = pointcloud_input.shape[0] / pointcloud_tgt.shape[0]

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()

# Print results
print('data completeness', eval_df_class.values[0])
