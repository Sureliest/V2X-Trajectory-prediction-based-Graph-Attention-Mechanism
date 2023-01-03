from torch.utils.data import Dataset, DataLoader
from data_process import read_data
import numpy as np
import torch


class ngsimDataset(Dataset):

    def __init__(self, file_path, enc_size=64, is_train=False):
        self.file_path = file_path
        self.enc_size = enc_size
        [self.all_features, self.all_neighbour, self.all_xy] = read_data()
        # features = (N, V, T, F)

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        # target_id = self.all_xy[idx, 0]
        # frame_id = self.all_xy[idx, 1]
        neighbours = []
        feature_matrix = self.all_features[idx]
        target__hist = feature_matrix[0, 0:30]
        target_fut = feature_matrix[0, 30:]
        for data in feature_matrix[1:]:
            if data[0, 0]:
                neighbours.append(data[0:30])

        return target__hist, target_fut, neighbours


def get_graph():
    pass

if __name__ == '__main__':
    trData = ngsimDataset('')
    data = DataLoader(dataset=trData, batch_size=64, shuffle=True)
