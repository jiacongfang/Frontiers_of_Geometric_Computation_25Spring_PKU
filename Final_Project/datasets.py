import os
import json
import h5py
import numpy as np
from termcolor import colored, cprint

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ShapeNetDataset(Dataset):    
    def __init__(self, info_file, dataroot, phase='train', cat='all', res=64, max_dataset_size=10000000, trunc_thres=0.2):
        """        
        Arguements:
            info_file (str): 'dataset_info_files/info-shapenet.json'
            dataroot (str).
            phase (str): 'train' or 'test'
            cat (str/list): 'all' or ["airplane", "chair", "speaker", "rifle", "sofa", "table"]
            res (int): resulution of SDF, use 64 here.
            max_dataset_size (int): maximum number of samples to load, default is float('inf') for no limit
            trunc_thres (float): truncation threshold for SDF values, where 0.0 means no truncation.
        """
        self.max_dataset_size = max_dataset_size
        self.res = res
        self.dataroot = dataroot
        self.trunc_thres = trunc_thres
        
        with open(info_file) as f:
            self.info = json.load(f)
            
        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}
        
        if cat == 'all':
            all_cats = self.info['all_cats']
        elif isinstance(cat, str):
            all_cats = [cat]
        elif isinstance(cat, list):
            all_cats = cat  
        else:
            raise ValueError("cat should be 'all', a string, or a list of strings.")
        
        self.model_list, self.cats_list = [], []
        
        for c in all_cats:
            synset = self.info['cats'][c]
            list_file = f'dataset_info_files/ShapeNet_filelists/{synset}_{phase}.lst'
            if not os.path.exists(list_file):
                print(f"Warning: no such filelists {list_file}")
                continue
                
            with open(list_file) as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')

                    path = f'{dataroot}/ShapeNet/SDF_v1/resolution_{self.res}/{synset}/{model_id}/ori_sample_grid.h5'
                    if os.path.exists(path):
                        model_list_s.append(path)
                
                self.model_list += model_list_s
                self.cats_list += [synset] * len(model_list_s)
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))
        
        rng = np.random.RandomState(0)
        p = rng.permutation(len(self.model_list))
        self.model_list = [self.model_list[i] for i in p]
        self.cats_list = [self.cats_list[i] for i in p]

        self.model_list = self.model_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')
        
        self.N = len(self.model_list)
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):
        synset = self.cats_list[index]
        sdf_h5_file = self.model_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        if self.trunc_thres != 0.0:
            sdf = torch.clamp(sdf, min=-self.trunc_thres, max=self.trunc_thres)
                
        ret = {
            'sdf': sdf,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_h5_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetSDFDataset'
        


class TextShapeNetDataset(Dataset):
    pass

if __name__ == "__main__":
    # test code for ShapeNetDataset
    dataset = ShapeNetDataset(
        info_file='./dataset_info_files/info-shapenet.json',
        dataroot='./data',
        phase='train',
        cat='all',
        res=64,
        trunc_thres=0.2
    )
    
    sample = dataset[0]
    print(f"SDF形状: {sample['sdf'].shape}")
    print(f"类别: {sample['cat_str']} ({sample['cat_id']})")
    print(f"文件路径: {sample['path']}")

    # test code for TextShapeNetDataset

