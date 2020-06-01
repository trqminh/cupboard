import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, obs_dim=14, act_dim=2, hidden_size=512):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + act_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        s = self.fc1(s)
        s = self.relu(s)
        s = torch.cat((s,a), axis=-1)
        s = self.fc2(s)
        s = self.relu(s)
        s = self.fc3(s)
        s = self.relu(s)

        return self.fc4(s)

if __name__ == '__main__':
    critic = Critic()
    s = torch.ones(14)
    a = torch.ones(2)
    print(critic(s,a))
