import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim=14, act_dim=2, hidden_size=512):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        s = torch.cat((s, a), dim=-1)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)

        return self.tanh(self.fc3(s))


# if __name__ == '__main__':
#     critic = Critic()
#     s = torch.ones(14)
#     a = torch.ones(2)
#     print(critic(s, a))
