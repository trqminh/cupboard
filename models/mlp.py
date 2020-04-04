import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(mlp, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return self.fc3(x)

