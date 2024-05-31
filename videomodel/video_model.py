import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoLeNet5(nn.Module):

    def __init__(self):

        super(VideoLeNet5, self).__init__()

        self.conv1 = nn.Conv3d(in_channels = 3, out_channels = 6, kernel_size = (5, 5, 5), stride = 1, padding = 0)
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2, 2), stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5,5,5), stride = 1, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = (2, 2, 2), stride = 2)
        self.fc3 = nn.Linear(400, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 29)


    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))

        x = x.view(-1, 400)

        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x