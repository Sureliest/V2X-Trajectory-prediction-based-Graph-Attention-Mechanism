from torch.utils.data import Dataset, DataLoader
from data_process import read_data
import numpy as np
import torch

class ngsimDataset(Dataset):

    def __init__(self, file_path, enc_size = 64, is_train = False):
        self.file_path = file_path #(T, V, N, F)
        self.enc_size = enc_size
        [self.all_features, self.all_neighbour, self.all_xy] = read_data()

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        self.vehicle_id = self.all_xy[idx, 0]
        self.frame_id = self.all_xy[idx, 1]
        neighbours = []
        all_featurs = []
        all_adjency = []