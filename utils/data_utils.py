import numpy as np
from sklearn.model_selection import train_test_split
import torch

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

def get_data_tag(faultP=[0, 1], desample=512):
    filename = [f"./data/processed/fault_probability_{i}.mat" for i in [15, 25, 50, 75, 100]]
    data = None
    tag = None
    for item in faultP:
        dataset = loadmat(filename[item])
        temp_data = dataset["dataset"]
        temp_tag = dataset["datatag"].flatten()
        step = int(temp_data.shape[0] / desample)
        temp_data = temp_data[::step,:,:]
        data = temp_data if data is None else np.concatenate((data, temp_data), axis=2)
        tag = temp_tag if tag is None else np.concatenate((tag, temp_tag), axis=0)
    data = np.transpose(data, (2, 1, 0))
    return data, tag


class MyDataset(Dataset):
    def __init__(self, arr, labels):
        self.data = arr
        self.tag = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index,...], self.tag[index]
    

def get_dataloader(data, labels):
    data_tensor = torch.from_numpy(data).float()
    tags_tensor = torch.from_numpy(labels)
    data_train, data_tmp, tag_train, tag_tmp = train_test_split(data_tensor, tags_tensor, test_size=0.3, random_state=22)
    data_val, data_test, tag_val, tag_test = train_test_split(data_tmp, tag_tmp, test_size=0.4, random_state=22)
    train_dataset = MyDataset(data_train, tag_train)
    val_dataset = MyDataset(data_val, tag_val)
    test_dataset = MyDataset(data_test, tag_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    print('Training set size:',len(train_dataset))
    print('Validation set size:',len(val_dataset))
    print('Test set size:',len(test_dataset))
    return train_loader, test_loader, val_loader


if __name__ == '__main__':
    
    data, tag = get_data_tag()
    print(data.shape, tag.shape)
    train_loader, test_loader, val_loader = get_dataloader(data, tag)