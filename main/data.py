from torch.utils.data import Dataset
import scipy as sci
import os
import torch

class DCO_dataset(Dataset):
    def __init__(self, mode='train', data_dir='../DCO_data/', normalize=True):
        super(DCO_dataset, self).__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.normalize = normalize
        if self.mode == 'train':
            mat_data = sci.io.loadmat(os.path.join(self.data_dir, 'train_data.mat'))
            self.E = mat_data['E_train']
            self.r = mat_data['r_train']
            self.curl = mat_data['curl_train']  # (Nx, Ny, Nz, 3, n_samples)
        else:
            mat_data = sci.io.loadmat(os.path.join(self.data_dir, 'test_data.mat'))
            self.E = mat_data['E_test']
            self.r = mat_data['r_test']
            self.curl = mat_data['curl_test']
        self.E = torch.from_numpy(self.E).permute(4, 3, 0, 1, 2).float()    # (n_samples, 3, Nx, Ny, Nz)
        self.r = torch.from_numpy(self.r).permute(4, 3, 0, 1, 2).float()
        self.curl = torch.from_numpy(self.curl).permute(4, 3, 0, 1, 2).float() 
        self.n_samples = self.E.shape[0]

        print(f"{mode} 数据集: {self.n_samples} 个样本")
        print(f"  E shape: {self.E.shape}")
        print(f"  curl shape: {self.curl.shape}")
        print(f"  r shape: {self.r.shape}")

        if self.normalize and self.mode == 'train':
            self.E_mean = torch.mean(self.E, dim=(1,2,3,4), keepdim=True)
            self.E_std = torch.std(self.E, dim=(1,2,3,4), keepdim=True)
            self.r_mean = torch.mean(self.r, dim=(1,2,3,4), keepdim=True)
            self.r_std = torch.std(self.r, dim=(1,2,3,4), keepdim=True)
            self.E = (self.E - self.E_mean) / (self.E_std + 1e-7)
            self.r = (self.r - self.r_mean) / (self.r_std + 1e-7)
            self.curl = self.curl * (self.r_std + 1e-7) / (self.E_std + 1e-7)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.E[idx], self.r[idx], self.curl[idx]