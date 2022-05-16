import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MovingMnist(Dataset):

    def __init__(self, data_dir, train=False):
        super(MovingMnist, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.data = self.train_eval_split()
        self.n_seq = self.data.shape[1]

    def __len__(self):
        return self.n_seq

    def __getitem__(self, item):
        input_seq = self.data[:, item:item+1, :, :] / 255

        return input_seq

    def train_eval_split(self):
        if self.train:
            data = np.load(self.data_dir)[:, 0:8000, :, :]
        else:
            data = np.load(self.data_dir)[:, 8000:10000, :, :]

        return data



