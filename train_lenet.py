import torch
import torch.optim as optim
import torch.nn as nn

from models import LeNet5
from dataProcessing import *
from workflow_functions import *


# set device to gpu if available and cpu if not
G_DEVICE            = torch.device("cuda") or torch.device("cpu")
G_BATCH_SIZE        = 128
G_NUM_WORKERS       = 16
G_TEST_IMG_PATH     = "ASL_Data/asl_alphabet_test/asl_alphabet_test"
G_TRAIN_IMG_PATH    = "ASL_Data/asl_alphabet_train/asl_alphabet_train"

# load data for LeNet
train_loader, test_loader = split_dataloaders(G_TRAIN_IMG_PATH, train_size = 0.8, batch_size = G_BATCH_SIZE, num_workers = G_NUM_WORKERS, transform = train_transform_le)

# some parameters for model training
learning_rate = 0.001
epochs = 30

# places to store historical metrics
lenet_train_losses = []; lenet_test_losses = []; 
lenet_train_accuracies = []; lenet_test_accuracies = []

# Perform workflow with LeNet
lenet = LeNet5().to(G_DEVICE)
optimizer = optim.Adam(lenet.parameters(), lr = learning_rate)
loss_criteria = nn.CrossEntropyLoss().to(G_DEVICE)

# training loop
for epoch in range(epochs):
    # get metrics
    train_loss, test_loss, train_accuracy, test_accuracy = train(lenet, train_loader, test_loader, optimizer, loss_criteria, epoch, device = G_DEVICE)

    # populate arrays containing metric by epoch
    lenet_train_losses.append(train_loss)
    lenet_test_losses.append(test_loss)
    lenet_train_accuracies.append(train_accuracy)
    lenet_test_accuracies.append(test_accuracy)

# save the LeNet model weights
save_model(lenet, "LeNet30Epoch.pth")

# save and show the plot for LeNet
make_plots("LeNet30EpochHistory.png", lenet_train_losses, lenet_test_losses, lenet_train_accuracies, lenet_test_accuracies)

# confusion matrix for LeNet
create_confusion_matrix("LeNet30EpochConfusion.png", lenet, test_loader, device = G_DEVICE)