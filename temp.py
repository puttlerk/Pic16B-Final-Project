import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import prune

from models import AlexNet
from dataProcessing import *
from workflow_functions import *


 # get the trained model weights
alex = load_model("AlexNet30Epoch.pth")

# prune the model because it is too big for github without paying for LFS
parameters_to_prune = (
    (alex.conv1, "weight"),
    (alex.conv2, "weight"),
    (alex.conv3, "weight"),
    (alex.conv4, "weight"),
    (alex.conv5, "weight"),
    (alex.fc1, "weight"),
    (alex.fc2, "weight"),
    (alex.fc3, "weight")
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method = prune.L1Unstructured,
    amount = 0.8
)
prune.remove(alex.conv1, "weight"),
prune.remove(alex.conv2, "weight"),
prune.remove(alex.conv3, "weight"),
prune.remove(alex.conv4, "weight"),
prune.remove(alex.conv5, "weight"),
prune.remove(alex.fc1, "weight"),
prune.remove(alex.fc2, "weight"),
prune.remove(alex.fc3, "weight")

save_model(alex, "AlexNet30Epoch.pth")