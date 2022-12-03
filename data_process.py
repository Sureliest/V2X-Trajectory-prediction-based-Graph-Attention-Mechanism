import numpy as np
import os

data_root = './Data/'
i80_path = 'i80/'
us101_path = 'us101/'
write_root = './Data/frame/i80/'

History_frames = 30  # 3 second * 10 frame
Future_frames = 30  # 3 second * 10 frame
Total_frames = History_frames + Future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
Max_num_object = 120  # maximum number of observed objects is 70
Neighbor_distance = 10  # meter
Delta = 0.3
data_list = [0, 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def unitConversion(dataset):
    # 单位转换
    ft_to_m = 0.3048  # 英尺转米
    dataset['Global_Time'] = dataset['Global_Time'] / 100
    for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_Length", "v_Width"]:
        dataset[strs] = dataset[strs] * ft_to_m
    dataset["v_Vel"] = dataset["v_Vel"] * ft_to_m * 3.6


def get_neighbour(now_dict, index, neighbour_distance):
    pass


def get_frame_instance_dict(file_path):
    pass


def process_data(now_dict, start_index, end_index, delta):
    pass


def generate_data(delta):
    train_data = []
    test_data = []

    return train_data, test_data


def generate_file(file_path):
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
    target_index = list(set(all_data[:, 0]))  # cached vehicle list
    target_index = sorted(target_index)
    frame_id_index = list(set(all_data[:, 1]))
    frame_id_index = sorted(frame_id_index)
    # print('vehicle list:', target_index)
    print('vehicle length:{} ; frame length:{}'.format(len(target_index), len(frame_id_index)))
    frame_dict = []
    for self_id in target_index:
        frame = [self_id]
        for data in all_data:
            if data[0] == self_id:
                frame.append(data[1])
        frame_dict.append(frame)

    print(frame_dict)
    # generate data file by frame order
    # for frame_id in frame_id_index:
    #     now_dict = []
    #     for data in all_data:
    #         if data[1] == frame_id:
    #             now_dict.append([data[i] for i in data_list])
    #     write_path = os.path.join(write_root, '{}.txt'.format(int(frame_id)))
    #     print('generating {}th frame data file'.format(int(frame_id)))
    #     np.savetxt(write_path, np.float_(now_dict), fmt='%f', delimiter=',')

def get_vehicle_arrange(file_name):
    print(file_name)
    with open(file_name, 'r') as reader:
        content = [x.strip().split(" ") for x in reader.readlines()]
        print(content)


if __name__ == '__main__':
    # print('Generating train/test data....:')
    # data_file_path = os.path.join(data_root, i80_path, 'i80-1600-1615.txt')
    # generate_file(data_file_path)
    # print('Exited program')

    file_path = '../Data/frame/i80/'
    file_list = os.listdir(file_path)
    max_x, max_y = 0, 0
    for file_name in file_list:
        file_name = file_path + file_name
        get_vehicle_arrange(file_name)
