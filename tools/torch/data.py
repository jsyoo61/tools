import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

class ProxyDataset(D.Dataset):
    """
    ProxyDataset that does not copy the data of the original dataset.
    Useful for train/validation/test split without modifying the data
    (e.g. validation/test data is not preprocessed using train data)

    Parameters
    ----------
    dataset : torch.utils.data.Dataset object
        The original dataset
    idxs : list
        List of indices to use for the proxy dataset
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

        # Copy attributes from the original dataset
        keys = [key for key in dir(dataset) if key.startswith('__') is False]
        for key in keys:
            setattr(self, key, getattr(dataset, key))

    def __repr__(self):
        return f'ProxyDataset({self.dataset}, len: {len(self)}/{len(self.dataset)}({len(self)/len(self.dataset)*1e2:.0f}%))'

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

    def __len__(self):
        return len(self.idxs)

def get_x_all(dataset):
    if hasattr(dataset, 'get_x_all'):
        return dataset.get_x_all()
    else:
        data_all = [data for data in dataset]
        if type(data_all[0]) == tuple: # More than 1 return value
            data_all = [data[0] for data in data_all]
        elif type(data_all[0]) == dict: # Dictionary return value
            data_all = [data['x'] for data in data_all]
        else: # 1 return value
            assert type(data_all[0]) == torch.Tensor

        return torch.stack(data_all, dim=0)

def get_y_all(dataset):
    if hasattr(dataset, 'get_y_all'):
        return dataset.get_y_all()
    else:
        data_all = [data for data in dataset]
        if type(data_all[0]) == tuple: # More than 1 return value
            data_all = [data[1] for data in data_all]
        elif type(data_all[0]) == dict: # Dictionary return value
            data_all = [data['y'] for data in data_all]
        else: # 1 return value
            raise Exception('No y data (2nd argument) found in the dataset')

        return torch.stack(data_all, dim=0)

def get_all(dataset):
    if hasattr(dataset, 'get_all'):
        return dataset.get_all()
    else:
        data_all = [data for data in dataset]
        if type(data_all[0]) == tuple: # More than 1 return value
            n_tuple = len(data_all[0])
            tensors = tuple([torch.stack([data[i] for data in data_all], dim=0) for i in range(n_tuple)])
        elif type(data_all[0]) == dict: # Dictionary return value
            keys = data_all[0].keys()
            tensors = {key: torch.stack([data[key] for data in data_all], dim=0) for key in keys}
        else: # 1 return value
            assert type(data_all[0]) == torch.Tensor
            tensors = torch.stack(data_all, dim=0)

    return tensors
