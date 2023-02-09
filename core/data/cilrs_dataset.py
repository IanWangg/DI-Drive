# import os
# import numpy as np
# from typing import Any, Dict
# import torch
# from torch.utils.data import Dataset
# 
# from core.utils.others.image_helper import read_image
# 
# 
# class CILRSDataset(Dataset):
# 
#     def __init__(self, root_dir: str, transform: bool = False, preloads: str = None) -> None:
#         self._root_dir = root_dir
#         self._transform = transform
# 
#         preload_file = preloads
#         if preload_file is not None:
#             print('[DATASET] Loading from NPY')
#             self._sensor_data_names, self._measurements = np.load(preload_file, allow_pickle=True)
# 
#     def __len__(self) -> int:
#         return len(self._sensor_data_names)
# 
#     def __getitem__(self, index: int) -> Any:
#         img_path = os.path.join(self._root_dir, self._sensor_data_names[index])
#         img = read_image(img_path)
#         if self._transform:
#             img = img.transpose(2, 0, 1)
#             img = img / 255.
#         img = img.astype(np.float32)
#         img = torch.from_numpy(img).type(torch.FloatTensor)
# 
#         measurements = self._measurements[index].copy()
#         data = dict()
#         data['rgb'] = img
#         for k, v in measurements.items():
#             v = torch.from_numpy(np.asanyarray([v])).type(torch.FloatTensor)
#             data[k] = v
#         return data
import os
import numpy as np
from typing import Any, Dict
import torch
from torch.utils.data import Dataset

from core.utils.others.image_helper import read_image


class CILRSDataset(Dataset):

    def __init__(self, root_dir: str, transform: bool = False, preloads: str = None, shrink: str = 'ratio:1') -> None:
        if reward:
            assert 'size' in shrink or shrink=='ratio:1', 'Reward enabled, needs terminal states'

        self._root_dir = root_dir
        self._transform = transform

        self.shrink_type, self.shrink_value = shrink.split(':')
        self.shrink_value = int(self.shrink_value)
        if self.shrink_type == 'ratio':
            assert self.shrink_value >= 1, 'Shrink ratio is at most 1'
        elif self.shrink_type == 'size':
            assert self.shrink_value >= int(4e3), 'Shrink size is at least 4e3'
        else:
            raise NotImplemented

        preload_file = preloads
        if preload_file is not None:
            print('[DATASET] Loading from NPY')
            self._sensor_data_names, self._measurements = np.load(preload_file, allow_pickle=True)

    def __len__(self) -> int:
        if self.shrink_type == 'ratio':
            return len(self._sensor_data_names) // self.shrink_value
        else:
            return min(len(self._sensor_data_names), self.shrink_value)

    def __getitem__(self, index: int) -> Any:
        img_path = os.path.join(self._root_dir, self._sensor_data_names[index])
        img = read_image(img_path)
        if self._transform:
            img = img.transpose(2, 0, 1)
            img = img / 255.
        img = img.astype(np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        measurements = self._measurements[index].copy()
        data = dict()
        data['rgb'] = img
        for k, v in measurements.items():
            v = torch.from_numpy(np.asanyarray([v])).type(torch.FloatTensor)
            data[k] = v
        return data

if __name__ == '__main__':
    dataset = CILRSDataset(
        root_dir='/home/ywang3/workplace/datasets_train/datasets_train/cilrs_datasets_train',
        transform=True,
        preloads='/home/ywang3/workplace/_preloads/cilrs_datasets_train.npy',
    )
    item = dataset[0]
    for k in item:
        print(k)
