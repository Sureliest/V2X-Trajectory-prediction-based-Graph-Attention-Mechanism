from torch.utils.data import Dataset, DataLoader
from data_process import read_data
import numpy as np
import torch


class ngsimDataset(Dataset):

    def __init__(self, file_path, enc_size=64, is_train=False):
        self.file_path = file_path
        self.enc_size = enc_size
        self.all_features, self.all_neighbour, self.all_xy = read_data(file_path)
        # features = (N, V, T, F)

    def __len__(self):
        return len(self.all_xy)

    def __getitem__(self, idx):
        now_target = self.all_xy[idx, 0]
        now_feature_matrix = self.all_features[idx]
        now_hist = now_feature_matrix[:, 0:30]
        target_fut = now_feature_matrix[0, 30:]
        # for data in now_feature_matrix[1:]:
        #     if data[0, 0]:
        #         neighbours.append(data[0:30])
        return now_hist, target_fut, now_target, self.all_neighbour[idx], self.all_xy[idx]




if __name__ == '__main__':
    trData = ngsimDataset('../dataset/TrainSet/i80_1_3s.pkl')
    trainData = DataLoader(dataset=trData, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
    for i, data in enumerate(trainData):
        all_hist, fut, v_id, neighbours, xy = data
        print('{}, hist:\n{}\nfut:\n{}'.format(i, all_hist, fut))
        print(all_hist.shape, fut.shape, neighbours.shape, xy.shape)
