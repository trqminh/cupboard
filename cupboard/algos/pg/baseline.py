import torch
import torch.nn as nn

class baseline(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(baseline, self).__init__()

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

