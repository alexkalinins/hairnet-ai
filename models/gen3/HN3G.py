import torch
import torch.nn as nn
import torch.nn.functional as F

# gen3 SINGULAR VALUE
class HairNetHN3G(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.conv4 = nn.Conv2d(128, 128, 5, 1)

        x = torch.rand(200, 200).view(-1, 1, 200, 200)
        self._lin_shape = None
        self._conv_pass(x)

        self.fc1 = nn.Linear(self._lin_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)  # we need 1 output parameter

    def _conv_pass(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))

        if self._lin_shape is None:
            self._lin_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self._conv_pass(x)
        x = x.view(-1, self._lin_shape)
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = F.softplus(self.fc4(x))
        x = F.softplus(self.fc5(x))

        return x

