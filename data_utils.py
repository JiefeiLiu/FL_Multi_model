import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Class to load custom dataset
class CustomDataset(Dataset):

    def __init__(self, X_data, y_data, nn_type):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        if nn_type == "MLP":
            self.y = torch.tensor(y_data, dtype=torch.float32)    # Binary classification
        elif nn_type == "MLP_Mult":
            self.y = torch.tensor(y_data, dtype=torch.long)         # Multi class classification
        self.length = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length