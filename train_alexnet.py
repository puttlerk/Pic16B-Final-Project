import torch
import torch.optim as optim
import torch.nn as nn

from models import AlexNet
from dataProcessing import *
from workflow_functions import *


# set device to gpu if available and cpu if not
G_DEVICE            = torch.device("cuda") or torch.device("cpu")
G_BATCH_SIZE        = 128
G_NUM_WORKERS       = 16
G_TEST_IMG_PATH     = "ASL_Data/asl_alphabet_test/asl_alphabet_test"
G_TRAIN_IMG_PATH    = "ASL_Data/asl_alphabet_train/asl_alphabet_train"

# load data for AlexNet
train_loader, test_loader = split_dataloaders(G_TRAIN_IMG_PATH, train_size = 0.8, batch_size = G_BATCH_SIZE, num_workers = G_NUM_WORKERS, transform = train_transform_alex)

# some parameters for training
learning_rate = 0.001
epochs = 30

# places to store historical metrics
alex_train_losses = []; alex_test_losses = []; 
alex_train_accuracies = []; alex_test_accuracies = []

# Perform workflow with AlexNet
alex = AlexNet().to(G_DEVICE)
optimizer = optim.Adam(alex.parameters(), lr = learning_rate)
loss_criteria = nn.CrossEntropyLoss().to(G_DEVICE)

print(alex)

# train AlexNet
for epoch in range(epochs):
    # get metrics
    train_loss, test_loss, train_accuracy, test_accuracy = train(alex, train_loader, test_loader, optimizer, loss_criteria, epoch, device = G_DEVICE)

    # populate arrays containing metric by epoch
    alex_train_losses.append(train_loss)
    alex_test_losses.append(test_loss)
    alex_train_accuracies.append(train_accuracy)
    alex_test_accuracies.append(test_accuracy)

# save the AlexNet model weights
save_model(alex, "AlexNet30Epoch.pth")

# save and show the plot for AlexNet
make_plots("AlexNet30EpochHistory.png", alex_train_losses, alex_test_losses, alex_train_accuracies, alex_test_accuracies)

# confusion matrix for AlexNet
create_confusion_matrix("AlexNet30EpochConfusion.png", alex, test_loader, device = G_DEVICE)