import os
import numpy as np
from .tools import load_pickle

def sort_load(data_dir, load_func = None):
    '''load all data in the specified directory, in a sorted way
    if load_func == None, defaults to pickle'''
    if load_func == None:
        load_func = load_pickle

    file_list = sorted(os.listdir(data_dir))
    loaded = list()

    for file in file_list:
        data_dir_1 = os.path.join(data_dir, file)
        data = load_func(data_dir_1)
        loaded.append(data)

    return loaded

def sample_train_data(dataset_A, dataset_B,ppgset_A,ppgset_B, n_frames=128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_ppg_A = list()
    train_data_ppg_B = list()
    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        data_ppg_A = ppgset_A[idx_A]
        frames_A_total = data_A.shape[1]
        frames_A_ppg_total = data_ppg_A.shape[1]
        #print(frames_A_total)
        #print(frames_A_ppg_total)
        #print(min([frames_A_total,frames_A_ppg_total]))
        assert min([frames_A_total,frames_A_ppg_total]) >= n_frames
        start_A = np.random.randint(min([frames_A_total,frames_A_ppg_total]) - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:, start_A:end_A])
        train_data_ppg_A.append(data_ppg_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        data_ppg_B = ppgset_B[idx_B]
        frames_B_total = data_ppg_B.shape[1]
        frames_B_ppg_total = data_ppg_B.shape[1]
        #print(min([frames_B_total,frames_B_ppg_total]))
        assert min([frames_B_total,frames_B_ppg_total]) >= n_frames
        start_B = np.random.randint(min([frames_B_total,frames_B_ppg_total]) - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:, start_B:end_B])
        train_data_ppg_B.append(data_ppg_B[:, start_B:end_B])
        #print(np.shape(data_B))#data_B
    #print(len(train_data_A))
    #print(len(train_data_B))
    train_data_ppg_A = np.array(train_data_ppg_A)
    train_data_ppg_B = np.array(train_data_ppg_B)
    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    #train_data_A = np.expand_dims(train_data_A, axis=-1)
    #train_data_B = np.expand_dims(train_data_B, axis=-1)
    return train_data_A, train_data_B,train_data_ppg_A,train_data_ppg_B

class Tree():
    '''Not implemented yet'''
    def __init__(self, data = None, parent = None):
        self.data = data

        self.parent = parent
        self.children = list()
        self.depth = 0

    def __call__(self):
        return self.data

    def __getitem__(self, key):
        pass

    def __setitem__(self, key):
        pass
    def __len__(self):
        pass
