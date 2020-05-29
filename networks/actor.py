import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim=14, hidden_size=512):
        super(Actor, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, s):
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.relu(s)
        s = self.fc3(s)
        s = self.relu(s)
        s = self.fc4(s)

        return torch.cat((self.sigmoid(s), self.tanh(s)))

if __name__ == '__main__':
    actor = Actor()
    s = torch.ones(14)
    print(actor(s))
