import sys
from collections import defaultdict
import itertools

import torch
import numpy as np


from dataset import PartialDatasetReader

def pad_tensor(vec, length, dim, pad_symbol):
    # vec.shape = [3, 4, 5]
    # length=7, dim=1 -> pad_size = (3, 7-4, 5)
    pad_size = list(vec.shape)
    pad_size[dim] = length - vec.shape[dim]
    answer = torch.cat([vec, torch.ones(*pad_size, dtype=torch.long) * pad_symbol], dim=dim)
    return answer

def pad_tensors(tensors, pad=0, dim=0, pad_inner=True):
    # дополняет тензоры из tensors до общей максимальной длины символом pad
    if isinstance(tensors[0], int):
        return torch.LongTensor(tensors)
    if dim > 0 and pad_inner:
        inner_tensors = [pad_tensors(tensor, pad=pad, dim=dim-1) for tensor in tensors]
        return pad_tensors(inner_tensors, pad=pad, dim=dim, pad_inner=False)
    tensor_type = torch.Tensor if "float" in str(getattr(tensors[0], "dtype", "")) else torch.LongTensor
    tensors = [tensor_type(tensor) for tensor in tensors]
    # print("== Shapes ==")
    # for tensor in tensors:
    #     print(tensor.shape)
    # print("== ==")
    L = max(tensor.shape[dim] for tensor in tensors)
    tensors = [pad_tensor(tensor, L, dim=dim, pad_symbol=pad) for tensor in tensors]
    return torch.stack(tensors, dim=0)

class FieldBatchDataLoader:

    def __init__(self, X, batch_size=32, sort_by_length=True,
                 length_field=None, pad_dim=None,
                 state=115, device="cpu"):
        self.X = X
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.length_field = length_field
        self.pad_dim = pad_dim or dict()
        self.device = device
        np.random.seed(state)

    def __len__(self):
        return (len(self.X)-1) // self.batch_size + 1

    def __iter__(self):
        if self.sort_by_length:
            # отсортировать индексы по длине объектов [1, ..., 32] -> [7, 4, 15, ...]
            # изменилось взятие длины из поля
            if self.length_field is not None:
                lengths = [len(x[self.length_field]) for x in self.X]
            else:
                lengths = [len(list(x.values())[0]) for x in self.X]
            order = np.argsort(lengths)
            # сгруппировать в батчи [7, 4, 15, 31, 3, ...] -> [[7, 4, 15, 31], [3, ...], ...]
            batched_order = np.array([order[start:start+self.batch_size]
                                      for start in range(0, len(self.X), self.batch_size)])
            # переупорядочить батчи случайно: [[3, 11, 21, 19], [27, ...], ..., [7, ...], ...]
            np.random.shuffle(batched_order[:-1])
            # собрать посл-ть индексов: -> [3, 11, 21, 19, 27, ...]
            self.order = np.fromiter(itertools.chain.from_iterable(batched_order), dtype=int)
        else:
            self.order = np.arange(len(self.X))
            np.random.shuffle(self.order)
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.X):
            raise StopIteration()
        end = min(self.idx + self.batch_size, len(self.X))
        indexes = [self.order[i] for i in range(self.idx, end)]
        batch = dict()
        batch_data = [self.X[i] for i in indexes]
        # перебираем все поля
        for field in self.X[indexes[0]]:
            data = [elem[field] for elem in batch_data]
            # print(field)
            # print(data)
            batch[field] = pad_tensors(data, dim=self.pad_dim.get(field, 0)).to(self.device)
        batch["indexes"] = indexes
        self.idx = end
        return batch


class MultiDatasetBatchLoader:
    
    def __init__(self, data, from_dataloaders=False,
                 batch_size=32, sort_by_length=True,
                 length_field=None, pad_dim=None, state=115,
                 device="cpu"):
        if from_dataloaders:
            self._dataloaders = data
        else:
            self._dataloaders = dict()
            self.batch_size = batch_size
            for key, curr_data in data.items():
                curr_batch_size = batch_size.get(key, 32) if isinstance(batch_size, dict) else batch_size
                curr_length_field = length_field.get(key) if isinstance(length_field, dict) else length_field
                self._dataloaders[key] = FieldBatchDataLoader(
                    curr_data, curr_batch_size,
                    sort_by_length=sort_by_length,
                    length_field=curr_length_field,
                    pad_dim=pad_dim,
                    state=state,
                    device=device
                )
    
    def __len__(self):
        return sum(len(dataloader) for key, dataloader in self._dataloaders.items())
    
    def __iter__(self):
        for key, dataloader in self._dataloaders.items():
            dataloader.__iter__()
        self.batch_order = [key for key, dataloader in self._dataloaders.items() for _ in range(len(dataloader))]
        np.random.shuffle(self.batch_order)
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= len(self.batch_order):
            raise StopIteration()
        key = self.batch_order[self.idx]
        self.idx += 1
        batch = self._dataloaders[key].__next__()
        batch["task"] = key
        return batch
    
class BucketDataLoader(MultiDatasetBatchLoader):
    
    def __init__(self, data, batch_size=32, sort_by_length=True,
                 length_field=None, pad_dim=None, state=115, device="cpu"):
        indexes_by_buckets = defaultdict(list)
        for i, elem in enumerate(data):
            bucket = elem.get("bucket")
            indexes_by_buckets[bucket].append(i)
        bucketed_data = {
            bucket: PartialDatasetReader(data, indexes) for bucket, indexes in indexes_by_buckets.items()
        }
        super().__init__(
            bucketed_data, batch_size=batch_size, sort_by_length=sort_by_length,
            length_field=length_field, pad_dim=pad_dim, state=state, device=device
        )
    
    