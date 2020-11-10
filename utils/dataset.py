#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yifan Zhu
# datetime:2020/10/1 23:52
# file: dataset.py
# software: PyCharm
import glob
import logging
import os

import torch
import yaml
from torch.utils import data
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        output_data: normalized data between [-0.5, 0.5]

    """

    pts = input_data
    size = pts.max(axis=0) - pts.min(axis=0)
    pts = pts / size.max()
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    output_data = pts

    return output_data


def create_partial_data(input_data, ind):
    """

    Args:
        input_data: complete input data
        ind: index of input data in 'train.lst'

    Returns:
        partial_data: synthetic partial data
        partial_data_ind: index of entries in the pratial data
    """
    if ind % 8 == 0:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] < 0, input_data[:, 1] < 0), input_data[:, 2] < 0))
        partial_data = input_data[partial_data_ind]
    elif ind % 8 == 1:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] < 0, input_data[:, 1] < 0), input_data[:, 2] > 0))
        partial_data = input_data[partial_data_ind]
    elif ind % 8 == 2:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] < 0, input_data[:, 1] > 0), input_data[:, 2] < 0))
        partial_data = input_data[partial_data_ind]
    elif ind % 8 == 3:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] < 0, input_data[:, 1] > 0), input_data[:, 2] > 0))
        partial_data = input_data[partial_data_ind]
    elif ind % 8 == 4:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] > 0, input_data[:, 1] < 0), input_data[:, 2] < 0))
        partial_data = input_data[partial_data_ind]
    elif ind % 8 == 5:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] > 0, input_data[:, 1] < 0), input_data[:, 2] > 0))
        partial_data = input_data[partial_data_ind]
    elif ind % 8 == 6:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] > 0, input_data[:, 1] > 0), input_data[:, 2] < 0))
        partial_data = input_data[partial_data_ind]
    else:
        partial_data_ind = np.where(
            np.logical_or(np.logical_or(input_data[:, 0] > 0, input_data[:, 1] > 0), input_data[:, 2] > 0))
        partial_data = input_data[partial_data_ind]

    return partial_data, partial_data_ind


def create_partial_data_with_cutting_plane(input_data, ind, data_completeness=0.5):
    """

    Args:
        input_data: complete input data
        ind: index of input data in 'train.lst'
        data_completeness: threshold of data completeness

    Returns:
        partial_data: synthetic partial data
        partial_data_ind: index of entries in the pratial data
    """

    r = R.from_euler('zxy', R.random(num=1, random_state=ind).as_euler('zxy', degrees=True), degrees=True)
    data_rot = r.apply(input_data)
    rs = np.random.RandomState(ind)
    offset = rs.rand() - 0.5
    partial_data_ind = np.where(data_rot[:, 1] > offset)
    selected = len(partial_data_ind[0]) / len(input_data)
    while selected > data_completeness or selected < data_completeness - 0.1:
        ind += 1
        rs = np.random.RandomState(ind)
        offset = rs.rand() - 0.5
        partial_data_ind = np.where(data_rot[:, 1] > offset)
        selected = len(partial_data_ind[0]) / len(input_data)
    partial_data = input_data[partial_data_ind]

    return partial_data, partial_data_ind


class ShapenetDataset(data.Dataset):
    """
    shapenet dataset class
    modified from https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/data/core.py
    """

    def __init__(self, dataset_folder, fields, categories=None, split=None, points_batch=128 ** 2, with_normals=False,
                 partial_input=False, data_completeness=1, data_sparsity=1, evaluation=False):
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.points_batch = points_batch
        self.with_normals = with_normals
        self.partial_input = partial_input
        self.eval = evaluation
        self.data_completeness = data_completeness
        self.data_sparsity = data_sparsity

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
            shapes_c = shapes_c[:1]

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
                    'Error occurred when loading field %s of model %s'
                    % (field_name, shape)
                )
                raise

            if isinstance(field_data, dict):
                pts = field_data['points'][::self.data_sparsity]
                if self.partial_input:
                    pts, partial_pts_ind = create_partial_data_with_cutting_plane(pts, idx, self.data_completeness)
                pts = torch.from_numpy(pts)
                if self.eval:
                    data['points_tgt'] = field_data['points']
                    data['points'] = pts
                else:
                    random_idx = torch.randperm(pts.shape[0])[:self.points_batch]
                    data['points'] = torch.index_select(pts, 0, random_idx)
                if self.with_normals:
                    normals = field_data['normals'][::self.data_sparsity]
                    if self.partial_input:
                        normals = normals[partial_pts_ind]
                    normals = torch.from_numpy(normals)
                    if self.eval:
                        data['normals_tgt'] = field_data['normals']
                        data['normals'] = normals
                    else:
                        data['normals'] = torch.index_select(normals, 0, random_idx)
            else:
                data[field_name] = field_data

        return data

    def get_model_dict(self, idx):
        return self.shapes[idx]


class KITTI360Dataset(data.Dataset):
    def __init__(self, dataset_folder, split, points_batch=200, evaluation=False):
        self.dataset_folder = dataset_folder
        self.split = split
        self.points_batch = points_batch
        self.evaluation = evaluation

        self.root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', self.dataset_folder,
                                     'data_3d_car_pointcloud')
        self.dirList = sorted(glob.glob(self.root_dir + '\*\*\*'), key=os.path.getmtime)

    def __len__(self):
        return len(self.dirList)

    def __getitem__(self, idx):
        data = {}
        shape_path = self.dirList[idx]
        pcd = normalize_data(np.load(shape_path).astype(np.float32))
        pcd = torch.from_numpy(pcd)
        if self.evaluation:
            data['points_tgt'] = pcd
        else:
            random_idx = torch.randperm(pcd.shape[0])[:self.points_batch]
            data['points'] = torch.index_select(pcd, 0, random_idx)

        return data
