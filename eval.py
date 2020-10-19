import argparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import torch
from utils import dataset
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud
from train import normalize_data

# hyper-parameters
checkpoint = '0300'
split = 'test'

pcl_dir = os.path.join('output', 'exp_2000', 'shapenet_car_zdim_256_partial_vae')

# create dataloader
DATA_PATH = 'data/ShapeNet'
fields = {
    'inputs': dataset.PointCloudField('pointcloud.npz')
}
test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=['02958343'],
                                       split=split, partial_input=False, evaluation=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

# Evaluator
evaluator = MeshEvaluator(n_points=100000)
eval_dicts = []

for it, data in enumerate(tqdm(test_loader)):
    # Evaluating mesh and pointcloud
    # Start row and put basic information inside
    eval_dict = {
        'idx': it,
    }
    eval_dicts.append(eval_dict)

    pointcloud_tgt = data['points'].squeeze(0).numpy()
    pointcloud_file = os.path.join(pcl_dir, 'mesh_{}_{}_{}.ply'.format(split, checkpoint, it))
    if os.path.exists(pointcloud_file):
        pointcloud = load_pointcloud(pointcloud_file).astype(np.float32)
        pointcloud = np.expand_dims(pointcloud, axis=0)
        pointcloud = normalize_data(pointcloud).squeeze(0)
        eval_dict_pcl = evaluator.eval_pointcloud(
            pointcloud, pointcloud_tgt)
        for k, v in eval_dict_pcl.items():
            eval_dict[k + ' (pcl)'] = v
    else:
        print('Warning: pointcloud does not exist: %s'% pointcloud_file)


out_file = os.path.join(pcl_dir, 'result', 'eval_input_full.pkl')
out_file_class = os.path.join(pcl_dir, 'result', 'eval_input.csv')

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
os.makedirs(pcl_dir + '/result', exist_ok=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.mean()
eval_df_class.to_csv(out_file_class)

# Print results
print(eval_df_class)
