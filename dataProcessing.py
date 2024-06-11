import os
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset, SubsetRandomSampler, Subset
from torchvision import transforms, datasets


# Define train transformations
train_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(degrees=(-15, 15), p=0.2),
    transforms.RandomPerspective(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define test transformations
test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Path to  training images
train_repo_path = "ASL_Data/asl_alphabet_train/asl_alphabet_train"

# Create PyTorch dataset using ImageFolder
full_dataset_train = datasets.ImageFolder(root=train_repo_path, transform=train_transform)
full_dataset_test = datasets.ImageFolder(root=train_repo_path, transform=test_transform)

# Calculates how many training examples should be in train and test datasets
total_size = len(full_dataset_train)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

# Gets random indices for test and training
train_indices = random.sample(range(total_size), train_size)
test_indices = list(set(range(total_size)) - set(train_indices))

# Splits the data into train_data and test_data
train_data = Subset(full_dataset_train, train_indices)
test_data = Subset(full_dataset_test, test_indices)


# A subset of training data
index = list(range(0, int(train_size * 0.1)))
fraction_data = torch.utils.data.Subset(train_data, index)

BATCH_SIZE = 256

# Dataloader for out train and test sets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# Dataloader for subset of training data that was used for hyperparameter tuning
fraction_loader = torch.utils.data.DataLoader(fraction_data, batch_size=BATCH_SIZE, shuffle=True)

"""
Might delete later...

class testImagesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # sorts list of file path names alphabetically
        self.images = sorted([os.path.join(root, img) for img in os.listdir(root)])
        # dict comprehension that assigns index as key to file path name
        self.labels = {os.path.basename(img): index for index, img in enumerate(self.images)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[os.path.basename(img_path)]
        return image, label

# creates our dataset and dataloader
test_dataset = testImagesDataset(root = "ASL_Data/asl_alphabet_test/asl_alphabet_test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size = 1) """
