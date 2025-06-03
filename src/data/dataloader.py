import numpy as np
from typing import List, Tuple, Iterator


class DataLoader:
    """DataLoader for batch processing of time series data"""

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            batch_x = []
            batch_y = []

            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)

            yield np.array(batch_x), np.array(batch_y)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size