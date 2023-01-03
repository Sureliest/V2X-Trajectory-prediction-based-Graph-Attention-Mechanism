import pickle

import numpy as np
import os
import math
from scipy import spatial

data_root = '../Data/'
i80_path = 'i80/'
us101_path = 'us101/'
write_root = './Data/frame/i80/'
vehicle_id_path = '../Data/frame/Vehicle_ID/'
frame_id_path = '../Data/frame/Frame_ID/'
i80_1 = 'i80-1600-1615.txt'
i80_2 = 'i80-1700-1715.txt'
i80_3 = 'i80-1715-1730.txt'
us101_1 = 'us101-0750-0805.txt'
us101_2 = 'us101-0805-0820.txt'
us101_3 = 'us101-0820-0835.txt'

History_frames = 30  # 3 second * 10 frame
Future_frames = 30  # 3 second * 10 frame 30/50
Total_frames = History_frames + Future_frames
Max_num_object = 30  # maximum number of observed objects
Neighbor_distance = 100  # meter
data_list = [0, 1, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


# 0:Vehicle_ID 1:Frame_ID 2:Total_Frames 3:Global_Time 4:Local_X 5:Local_Y 6:Global_X 7:Global_Y 8:v_Length 9:v_Width
# 10:v_Class 11:v_Vel 12:v_Acc 13:Lane_ID 14:Preceding 15:Following 16:Space_Headway 17:Time_Headway


def unitConversion(dataset):
    # 单位转换
    ft_to_m = 0.3048  # 英尺转米
    dataset['Global_Time'] = dataset['Global_Time'] / 100
    for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_Length", "v_Width"]:
        dataset[strs] = dataset[strs] * ft_to_m
    dataset["v_Vel"] = dataset["v_Vel"] * ft_to_m * 3.6


def compute_distance(list1, list2):
    ft_to_m = 0.3048
    return math.sqrt((list2[1] * ft_to_m - list1[1] * ft_to_m) ** 2 + (list2[0] * ft_to_m - list1[0] * ft_to_m) ** 2)


def get_neighbour(all_data, veh_ID, t_ID, neighbour_distance):
    neighbour_matrix = []
    neighbours = []
    target_xy = []
    neighbour_list = []
    for data in all_data:
        if data[0] == veh_ID and data[1] == t_ID:
            target_xy = data[4:6]
        elif data[0] != veh_ID and data[1] == t_ID:
            neighbours.append(data)
    for neighbour in neighbours:
        if compute_distance(neighbour[4:6], target_xy) <= neighbour_distance and neighbours:
            if len(neighbour_list) < Max_num_object - 1:
                neighbour_matrix.append(neighbour[data_list])
                neighbour_list.append(neighbour[0])
    # print(t_ID)
    # print('neighbours:{}, neighbour_matrix:{}'.format(neighbour_list, neighbour_matrix))
    length = len(neighbour_matrix)
    zero = np.zeros([Max_num_object - length - 1, len(data_list)])
    return np.array(neighbour_matrix + list(zero), dtype=float), neighbour_list


def get_feature_matrix(all_data, neighbours, veh_ID, t_ID):
    start_frame = int(t_ID) - History_frames + 1
    end_frame = int(t_ID) + Future_frames
    feature_matrix = np.zeros((Max_num_object, History_frames + Future_frames, len(data_list)))
    timelist = [time for time in range(start_frame, end_frame + 1)]
    for time_index, time in enumerate(timelist):
        now_dict = all_data[all_data[:, 0] == veh_ID]
        data = now_dict[now_dict[:, 1] == time]
        feature_matrix[0, time_index] = data[0][data_list]
        for n_index, n_id in enumerate(neighbours):
            # print(n_index, n_id)
            n_dict = all_data[all_data[:, 0] == n_id]
            n_data = n_dict[n_dict[:, 1] == time]
            if n_data.size > 0:
                feature_matrix[n_index + 1, time_index] = n_data[0][data_list]
    return list(feature_matrix)


def generate_data(all_data):
    all_features = []
    all_neighbours = []
    all_xy = []
    t_range = all_data[0, 2]
    # print(t_range)
    for data in all_data[1000000:1000300]:
        veID = data[0]
        t_ID = data[1]
        now_data = all_data[all_data[:, 0] == veID]
        if len(now_data[now_data[:, 1] < t_ID]) >= History_frames and \
                len(now_data[now_data[:, 1] > t_ID]) >= Future_frames:
            neighbour_matrix, neighbours = get_neighbour(all_data, veID, t_ID, Neighbor_distance)
            all_xy.append(data[data_list])
            all_neighbours.append(neighbour_matrix)
            print('neighbours', neighbours)
            matrix = get_feature_matrix(all_data, neighbours, veID, t_ID, )
            all_features.append(matrix)
    return np.array(all_features), np.array(all_neighbours), np.array(all_xy)


def generate_file(file_path):
    print(file_path)
    all_data = []
    # get all data from txt:
    with open(file_path, 'r') as reader:
        content = [x.strip().split(" ") for x in reader.readlines()]
        for column in content:
            vehicle_data = []
            for i in column:
                if i != '':
                    vehicle_data.append(float(i))
            all_data.append(vehicle_data)
    all_data = np.array(all_data)  # cached all data
    all_features, all_neighbour, all_xy = generate_data(all_data)
    # print('features[0]:', all_features[:, 0])
    # print('features[1]', all_features[:, 1])
    # print('features[2]', all_features[:, 2])
    print(all_features.shape, all_neighbour.shape, all_xy.shape)
    return all_features, all_neighbour, all_xy
    # print(all_data)
    # local_x = all_data[:, 4]
    # local_y = all_data[:, 5]
    # print(local_x, local_y)
    # min_x = min(local_x)
    # max_x = max(local_x)
    # min_y = min(local_y)
    # max_y = max(local_y)
    # return min_x*0.3048, min_y*0.3048, max_x*0.3048, max_y*0.3048

    # target_index = list(set(all_data[:, 0]))  # cached vehicle list
    # target_index = sorted(target_index)
    # frame_id_index = list(set(all_data[:, 1]))
    # frame_id_index = sorted(frame_id_index)
    # # print('vehicle list:', target_index)
    # print('vehicle length:{} ; frame length:{}'.format(len(target_index), len(frame_id_index)))
    # frame_dict = []
    # for self_id in target_index:
    #     frame = [self_id]
    #     for data in all_data:
    #         if data[0] == self_id:
    #             frame.append(data[1])
    #     frame_dict.append(frame)
    #
    # print(frame_dict)
    # generate data file by frame order
    # for frame_id in frame_id_index:
    #     now_dict = []
    #     for data in all_data:
    #         if data[1] == frame_id:
    #             now_dict.append([data[i] for i in data_list])
    #     write_path = os.path.join(write_root, '{}.txt'.format(int(frame_id)))
    #     print('generating {}th frame data file'.format(int(frame_id)))
    #     np.savetxt(write_path, np.float_(now_dict), fmt='%f', delimiter=',')


def save_data(features, neighbour, xy):
    save_path = 'save_data.pkl'
    with open(save_path, 'wb') as writer:
        pickle.dump([features, neighbour, xy], writer)


def read_data(file_path):
    with open(file_path, 'rb') as reader:
        [features, neighbour, xy] = pickle.load(reader)
    return features, neighbour, xy


if __name__ == '__main__':
    print('Generating train/test data....:')
    data_file_path = os.path.join(data_root, i80_path, i80_1)
    all_features, all_neighbour, all_xy = generate_file(data_file_path)
    save_data(all_features, all_neighbour, all_xy)
    # a = np.zeros((2, 3, 4))
    # # a[0, 0] = np.array([1, 1, 1, 1])
    # for i, num in enumerate(range(3)):
    #     a[0, i] = np.array([num, num+1, num+2, num+3])
    # print(a[a[:, :, 0] == 0 and a[:, :, 1] == 1])

    # data_file_path2 = os.path.join(data_root, i80_path, i80_2)
    # generate_file(data_file_path2)

    # data_file_path3 = os.path.join(data_root, i80_path, i80_3)
    # generate_file(data_file_path3)

    # min_x, min_y, max_x, max_y: [0.117 0.0 93.659 1757.488] feet
    # [0.0356616 0.0 28.547263200000003 535.6823424] meters  % meter = feet * 0.3048

    # a1 = [[1, 1], [2, 2]]
    # a2 = [[2, 1], [2, 5]]
    # print(spatial.distance.cdist(a1, a2))
