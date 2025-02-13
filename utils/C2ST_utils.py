import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

class LabeledDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, manifold):
        super(BinaryClassifier, self).__init__()
        if manifold.name == 'Special Orthogonal':
            input_size = input_size ** 2
        elif manifold.name == 'Unitary':
            input_size = 2 * input_size ** 2
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )
        self.dim = input_size

    def forward(self, x):
        if x.dtype == torch.complex64:
            x = torch.view_as_real(x)
        x = x.reshape(-1, self.dim)
        return self.model(x)