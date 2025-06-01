from typing import Iterator, Tuple, Optional
import torch
from ..core.tensor import Tensor

class DataLoader:
    """DataLoader for batching and shuffling time series data."""
    
    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        indices = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_x, batch_y = [], []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
                
            yield (Tensor(torch.stack(batch_x)), 
                  Tensor(torch.stack(batch_y)))
    
    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size