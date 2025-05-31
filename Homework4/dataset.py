import torch
import numpy as np
from torch.utils.data import Dataset
import os

class RecTrainDataset(Dataset):
    def __init__(self, data_root, dtype=torch.float32, sample_size=16384, device='cpu'):
        self.data_root = data_root
        self.object_name = os.listdir(data_root)
        self.dtype = dtype
        self.sample_size = sample_size
        self.device = device

    def load_sample_npz(self, npz_path):
        with np.load(npz_path) as data:
            points = data['points']
            grad = data['grad']
            sdf = data['sdf']
        
        points = torch.from_numpy(points)
        grad = torch.from_numpy(grad)
        sdf = torch.from_numpy(sdf)

        return points, grad, sdf
    
    def load_cloud_npz(self, npz_path):
        with np.load(npz_path) as data:
            points = data['points']
            normals = data['normals']
        
        points = torch.from_numpy(points)
        normals = torch.from_numpy(normals)
        sdf = torch.zeros(points.shape[0], dtype=self.dtype, device=self.device)  
        return points, normals, sdf


    def __len__(self):
        return len(self.object_name)
    
    def __getitem__(self, idx):
        cloud_npz_path = os.path.join(self.data_root, self.object_name[idx], 'pointcloud.npz')
        points, grad, sdf = self.load_cloud_npz(cloud_npz_path)

        random_idx = torch.randperm(points.shape[0])[:self.sample_size] 

        points = points[random_idx].to(device=self.device, dtype=self.dtype)
        grad = grad[random_idx].to(device=self.device, dtype=self.dtype)
        sdf = sdf[random_idx].to(device=self.device, dtype=self.dtype)

        return {
            'point': points,
            'grad': grad,
            'sdf': sdf,
            'object_name': self.object_name[idx]
        }


class RecTrainDataset_Mixed(Dataset):
    def __init__(self, data_root, dtype=torch.float32, sample_size=16384, device='cpu', ratio = 0.4):
        self.data_root = data_root
        self.object_name = os.listdir(data_root)
        self.dtype = dtype
        self.sample_size = sample_size
        self.device = device
        self.ratio = ratio  # Ratio of points from sample to total points   

    def load_sample_npz(self, npz_path):
        with np.load(npz_path) as data:
            points = data['points']
            grad = data['grad']
            sdf = data['sdf']
        
        points = torch.from_numpy(points)
        grad = torch.from_numpy(grad)
        sdf = torch.from_numpy(sdf)

        return points, grad, sdf
    
    def load_cloud_npz(self, npz_path):
        with np.load(npz_path) as data:
            points = data['points']
            normals = data['normals']
        
        points = torch.from_numpy(points)
        normals = torch.from_numpy(normals)
        sdf = torch.zeros(points.shape[0], dtype=self.dtype, device=self.device)  
        return points, normals, sdf


    def __len__(self):
        return len(self.object_name)
    
    def __getitem__(self, idx):
        sample_npz_path = os.path.join(self.data_root, self.object_name[idx], 'sdf.npz')
        sample_points, sample_grad, sample_sdf = self.load_sample_npz(sample_npz_path)

        cloud_npz_path = os.path.join(self.data_root, self.object_name[idx], 'pointcloud.npz')
        cloud_points, cloud_normals, cloud_sdf = self.load_cloud_npz(cloud_npz_path)

        sample_num = int(self.sample_size * self.ratio)
        sample_num_cloud = self.sample_size - sample_num

        random_idx_sample = torch.randperm(sample_points.shape[0])[:sample_num]
        random_idx_cloud = torch.randperm(cloud_points.shape[0])[:sample_num_cloud]
        points = torch.cat([sample_points[random_idx_sample], cloud_points[random_idx_cloud]], dim=0).to(device=self.device, dtype=self.dtype)
        grad = torch.cat([sample_grad[random_idx_sample], cloud_normals[random_idx_cloud]], dim=0).to(device=self.device, dtype=self.dtype)
        sdf = torch.cat([sample_sdf[random_idx_sample], cloud_sdf[random_idx_cloud]], dim=0).to(device=self.device, dtype=self.dtype)

        # suffle the points, grad, and sdf together
        random_idx = torch.randperm(points.shape[0])
        points = points[random_idx]
        grad = grad[random_idx]
        sdf = sdf[random_idx]

        return {
            'point': points,
            'grad': grad,
            'sdf': sdf,
            'object_name': self.object_name[idx]
        }


