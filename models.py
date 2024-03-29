import torch
from torch import nn


# MLP for Binary Classification
class MLP(nn.Module):
    def __init__(self, input_shape, num_units=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, 40)
        self.fc2 = nn.Linear(40, num_units)
        self.fc3 = nn.Linear(num_units, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# MLP for Multi-class Classification
class MLP_Mult(nn.Module):
    def __init__(self, input_shape, num_units=64, first_hidden=64, second_hidden=128, num_classes=11):
        super(MLP_Mult, self).__init__()
        self.fc1 = nn.Linear(input_shape, first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.fc3 = nn.Linear(second_hidden, num_units)
        self.fc4 = nn.Linear(num_units, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
