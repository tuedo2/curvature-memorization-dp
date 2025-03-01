import torch
from torch.utils.data import Dataset

class CorruptedDataset(Dataset):
    def __init__(self, base_dataset, transforms=None, corruption=None, corrupt_idx=None):
        self.dataset = base_dataset
        self.transforms = transforms
        self.corruption = corruption
        self.corrupt_idx = corrupt_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        if idx not in self.corrupt_idx:
            return img, label
        else:
            img = self.corruption(img)
            return img, label
        

class Zero(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img):
        return torch.zeros_like(img)