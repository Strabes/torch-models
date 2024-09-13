
import torch
from torch.utils.data import Dataset
from typing import Dict


class MulitModalDataset(Dataset):
    def __init__(
        self,
        data,
        dtype = torch.float64):
        """
        Parameters
        ----------
        data : Dict

        """
        self.data = data
        self.dtype = dtype

    def __len__(self):
        return len(self.data['numeric'])

    def __getitem__(self, idx):
        numeric = torch.from_numpy(self.data['numeric'][idx,:]).to(dtype=self.dtype)
        categorical = torch.from_numpy(self.data['categorical'][idx,:]).to(dtype=self.dtype)
        text = [v['input_ids'][idx,:] for k,v in self.data['text'].items()]
        return numeric, categorical, text

class CustomBatch:
    def __init__(self, data):
        data_collated = list(zip(*data))
        self.numeric = torch.stack(data_collated[0],0)
        self.categorical = torch.stack(data_collated[1],0)
        self.text = [torch.stack(x,0) for x in list(zip(*data_collated[2]))]

    def pin_memory(self):
        self.numeric = self.numeric.pin_memory()
        self.categorical = self.categorical.pin_memory()
        self.text = [i.pin_memory() for i in self.text]
        return self

def collate_wrapper(batch):
    return CustomBatch(batch)

class MulitModalDatasetWithTargets(MulitModalDataset):
    def __init__(self, data: Dict, dtype, scale_weights_by=None):
        super(MulitModalDatasetWithTargets, self).__init__(data, dtype)
        self.scale_weights_by = scale_weights_by
    
    def __getitem__(self, idx):
        target = torch.tensor(self.data["target"][idx], dtype=self.dtype)
        loss_weights = torch.tensor(self.data["loss_weights"][idx], dtype=self.dtype)
        if self.scale_weights_by:
            if loss_weights.sum() > 0:
                loss_weights = loss_weights * self.scale_weights_by / loss_weights.sum()
        numeric, categorical, text = super(MulitModalDatasetWithTargets, self).__getitem__(idx)
        return numeric, categorical, text, target, loss_weights

class CustomBatchWithTargets(CustomBatch):
    def __init__(self, data):
        super(CustomBatchWithTargets, self).__init__(data)
        data_collated = list(zip(*data))
        self.target = torch.stack(data_collated[3],0)
        self.loss_weights = torch.stack(data_collated[4],0)

    def pin_memory(self):
        super(CustomBatchWithTargets, self).pin_memory()
        self.target = self.target.pin_memory()
        self.loss_weights = self.loss_weights.pin_memory()
        return self

def collate_wrapper_with_targets(batch):
    return CustomBatchWithTargets(batch)