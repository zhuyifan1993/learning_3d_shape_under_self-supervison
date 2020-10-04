#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/10/1 23:52
# file: dataset.py
# software: PyCharm
import logging
import os

import torch
import yaml
from torch.utils import data
import numpy as np

logger = logging.getLogger(__name__)


class Field(object):
    """ Data fields class.
    """

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class PointCloudField(Field):
    """ Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    """

    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            'points': points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


def normalize_data(input_data):
    """

    Args:
        input_data: raw data, size=(point_samples, point_dimension)

    Returns:
        output_data: normalized data between [-1, 1]

    """

    pts = input_data
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = 2 * pts / size.max()
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    output_data = pts

    return output_data


class ShapenetDataset(data.Dataset):
    """
    shapenet dataset class
    modified from https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/data/core.py
    """

    def __init__(self, dataset_folder, fields, categories=None, split=None, points_batch=128 ** 2, with_normals=False):
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.points_batch = points_batch
        self.with_normals = with_normals

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories if os.path.isdir(os.path.join(dataset_folder, c))]

        # read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.FullLoader)
        else:
            self.metadata = {c: {'id': c, 'name': 'n/a'} for c in categories}

        # set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all shapes
        self.shapes = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                shapes_c = f.read().split('\n')

            # limit data length
            # shapes_c = shapes_c[:200]

            self.shapes += [
                {'category': c, 'shape': s}
                for s in shapes_c
            ]

    def __len__(self):

        return len(self.shapes)

    def __getitem__(self, idx):
        category = self.shapes[idx]['category']
        shape = self.shapes[idx]['shape']
        c_idx = self.metadata[category]['idx']

        shape_path = os.path.join(self.dataset_folder, category, shape)
        data = {}

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(shape_path, idx, c_idx)
            except Exception:
                logger.warning(
                    'Error occured when loading field %s of model %s'
                    % (field_name, shape)
                )
                raise

            if isinstance(field_data, dict):
                pts = torch.from_numpy(normalize_data(field_data['points']))
                random_idx = torch.randperm(pts.shape[0])[:self.points_batch]
                data['points'] = torch.index_select(pts, 0, random_idx)
                if self.with_normals:
                    normals = torch.from_numpy(field_data['normals'])
                    data['normals'] = torch.index_select(normals, 0, random_idx)
            else:
                data[field_name] = field_data

        return data
