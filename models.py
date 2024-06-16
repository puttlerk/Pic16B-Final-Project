import torch.nn as nn
import torch.nn.functional as F


# Classic model architecture of LeNet5
class LeNet5(nn.Module):

    def __init__(self):

        super(LeNet5,self).__init__()


        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6, kernel_size=(5,5), stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16, kernel_size=(5,5), stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc3 = nn.Linear(400,120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 29)


        self.conv1  = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = (5, 5), stride = 1, padding = 0)
        self.pool1  = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        self.conv2  = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = 1, padding = 0)
        self.pool2  = nn.MaxPool2d(kernel_size = (2, 2), stride = 2)
        self.fc3    = nn.Linear(400, 120)
        self.fc4    = nn.Linear(120, 84)
        self.fc5    = nn.Linear(84, 29)
        self.drop   = nn.Dropout(p = 0.5)


    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc3(x))

        x = self.drop(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Classic model architecture of AlexNet
class AlexNet(nn.Module):

    def __init__(self):

        super(AlexNet, self).__init__()

        self.conv1  = nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride = 4, padding = 0)
        self.pool1  = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2  = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
        self.pool2  = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3  = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv4  = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1)
        self.conv5  = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.pool3  = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        self.drop1  = nn.Dropout(p = 0.5)
        self.fc1    = nn.Linear(256 * 6 * 6, 4096)
        self.drop2  = nn.Dropout(p = 0.5)
        self.fc2    = nn.Linear(4096, 4096)
        self.fc3    = nn.Linear(4096, 29)

    def forward(self, input):
        input = F.relu(self.conv1(input))
        input = self.pool1(input)
        input = F.relu(self.conv2(input))
        input = self.pool2(input)
        input = F.relu(self.conv3(input))
        input = F.relu(self.conv4(input))
        input = F.relu(self.conv5(input))
        input = self.pool3(input)
        input = self.drop1(input)
        input = input.view(-1, 256 * 6 * 6)
        input = F.relu(self.fc1(input))
        input = self.drop2(input)
        input = F.relu(self.fc2(input))
        input = self.fc3(input)
        return input