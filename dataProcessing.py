import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


# Define transformations (TODO figure out proper transformations)
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Path to  training images
local_repo_path = "ASL_Data/asl_alphabet_train/asl_alphabet_train"


# Create PyTorch dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=local_repo_path, transform=transform)


# Calculates how many training examples should be in train and validation datasets
total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Splits the data into train_data and validation_data
train_data, val_data = random_split(train_dataset, [train_size, val_size])

print(type(val_data))

# Dataloader for out train and validation sets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

