import argparse
import numpy as np
import os

import trimesh
from tqdm import tqdm
import pandas as pd
import torch
from utils import dataset
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud

# hyper-parameters
checkpoint = 'final'
split = 'test'
partial_input = True
data_completeness = 0.7
data_sparsity = 100
eval_mesh = True
eval_pointcloud = True

output_dir = os.path.join('output', 'exp_partial', 'shapenet_car_zdim_256_04_100_pb300')

# create dataloader
DATA_PATH = 'data/ShapeNet'
fields = {
    'inputs': dataset.PointCloudField('pointcloud.npz')
}
test_dataset = dataset.ShapenetDataset(dataset_folder=DATA_PATH, fields=fields, categories=['02958343'],
                                       split=split, partial_input=partial_input, data_completeness=data_completeness,
                                       data_sparsity=data_sparsity, evaluation=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

# Evaluator
evaluator = MeshEvaluator(n_points=100000)
eval_dicts = []

for it, data in enumerate(tqdm(test_loader)):
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
    normal_tgt = data['normals_tgt'].squeeze(0).numpy()
    pointcloud_input = data['points'].squeeze(0).numpy()
    eval_dict['input/ground truth'] = pointcloud_input.shape[0] / pointcloud_tgt.shape[0]

    # Evaluate mesh
    if eval_mesh:
        mesh_file = os.path.join(output_dir,
                                 'mesh_{}_{}_{}_{}_{}.off'.format(split, data_completeness, data_sparsity, checkpoint,
                                                                  it))

        if os.path.exists(mesh_file):
            mesh = trimesh.load(mesh_file)
            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normal_tgt)
            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    if eval_pointcloud:
        pointcloud_file = os.path.join(output_dir,
                                       'mesh_{}_{}_{}_{}_{}.ply'.format(split, data_completeness, data_sparsity,
                                                                        checkpoint, it))
        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file).astype(np.float32)
            eval_dict_pcl = evaluator.eval_pointcloud(
                pointcloud, pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s' % pointcloud_file)

out_file = os.path.join(output_dir, 'result',
                        'eval_input_full_{}_{}_{}_{}.pkl'.format(split, data_completeness, data_sparsity, checkpoint))
out_file_class = os.path.join(output_dir, 'result',
                              'eval_input_{}_{}_{}_{}.csv'.format(split, data_completeness, data_sparsity, checkpoint))

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
os.makedirs(output_dir + '/result', exist_ok=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.loc['mean'] = eval_df_class.mean()
eval_df_class.to_csv(out_file_class)

# Print results
print(eval_df_class)
