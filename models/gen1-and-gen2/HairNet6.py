import torch
import torch.nn as nn
import torch.nn.functional as F


class HairNet6(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.fc1 = nn.Linear(200*200, 40000)
        self.fc2 = nn.Linear(40000, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 9)  # we need 9 parameters


    def forward(self, x):
        x = x.view(-1, 1, 200*200)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x

