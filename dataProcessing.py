import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets


# Define transformations
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Path to  training images
train_repo_path = "ASL_Data/asl_alphabet_train/asl_alphabet_train"

# Create PyTorch dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=train_repo_path, transform=transform)


# Calculates how many training examples should be in train and validation datasets
total_size = len(train_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Splits the data into train_data and validation_data
train_data, val_data = random_split(train_dataset, [train_size, val_size])


BATCH_SIZE = 256

# Dataloader for out train and validation sets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


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
test_loader = DataLoader(test_dataset, batch_size = 1)