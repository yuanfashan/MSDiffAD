import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import torch.utils.data as Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, intensity):
        self.input_ids = input_ids
        self.intensity = intensity

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.intensity[idx]

class MyDataSet3(Data.Dataset):
    def __init__(self, input_ids, intensity,res):
        self.input_ids = input_ids
        self.intensity = intensity
        self.res = res

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.intensity[idx], self.res[idx]

