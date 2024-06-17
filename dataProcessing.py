import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# This file creates train and val loaders for our two pre processing pipelines
# Needs dataset downloaded and ASL_Data folder in the same directory

# Define train transformations
# LeNet
train_transform_le = transforms.Compose([
    # Resize the image to expected the size for LeNet
    transforms.Resize((32, 32)),
    # Data Augmentation: RandomRotations
    transforms.RandomRotation(degrees=15),
    # Data Augmentation: Color jitter
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Conver to tensor
    transforms.ToTensor(),

    # Normalize
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# AlexNet
train_transform_alex = transforms.Compose([
    # Resize the image to expected the size for AlexNet
    transforms.Resize((227, 227)),
    # Data Augmentation: RandomRotations
    transforms.RandomRotation(degrees=15),
    # Data Augmentation: Color jitter
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Conver to tensor
    transforms.ToTensor(),
    # Normalize

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define test transformations
# Lenet
test_transform_le = transforms.Compose([
    # Resize the image to expected the size for LeNet
    transforms.Resize((32, 32)),
    # Convert to tensor
    transforms.ToTensor(),

    # Normalize
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# AlexNet
test_transform_alex = transforms.Compose([
    # Resize the image to the expected size for AlexNet
    transforms.Resize((227, 227)),
    # Convert to tensor
    transforms.ToTensor(),
    # Normalize
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def build_dataloader(dir, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = False, transform = None):
    """
    Split the data in the directory dir into training and validation subsets

    @params:
        dir: string, path to the directory containing the image data to load
        batch_size: int greater than 0, batch_size
        shuffle: bool, whether data is shuffled when building dataloader
        num_workers: int greater than 0, num_workers in dataloader
        pin_memory: bool, pin_memory in dataloader
        transform: pytorch transform, transform in dataloader

    @returns:
        loader, dataloader representing the data 
    """
    # Create PyTorch dataset using ImageFolder for LeNet
    dataset = datasets.ImageFolder(root = dir, transform = transform)

    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)


def split_dataloaders(dir, train_size = 0.8, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = False, transform = None):
    """
    Split the data in the directory dir into training and validation subsets

    @params:
        dir: string, path to the directory containing the image data to load
        train_size: float between 0 and 1, size of the training set
        batch_size: int greater than 0, batch_size
        shuffle: bool, whether data is shuffled when building dataloader
        num_workers: int greater than 0, num_workers in dataloader
        pin_memory: bool, pin_memory in dataloader
        transform: pytorch transform, transform in dataloader

    @returns:
        (train_loader, val_loader) dataloaders representing the training and validation data 
    """
    # Create PyTorch dataset using ImageFolder for LeNet
    dataset = datasets.ImageFolder(root = dir, transform = transform)

    # Split the data into training and testing subsets
    train_data, test_data = random_split(dataset, [train_size, 1 - train_size])

    # Instantiate the train loader
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)

    # Instantiate the test loader
    val_loader = DataLoader(test_data, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)

    return (train_loader, val_loader)


# save some dataloaders
# set device to gpu if available and cpu if not
G_BATCH_SIZE        = 128
G_NUM_WORKERS       = 16
G_TRAIN_IMG_PATH    = "ASL_Data/asl_alphabet_train/asl_alphabet_train"

# load data for LeNet
le_train_loader, le_val_loader = split_dataloaders(G_TRAIN_IMG_PATH, train_size = 0.8, batch_size = G_BATCH_SIZE, num_workers = G_NUM_WORKERS, transform = train_transform_le)
torch.save(le_train_loader, 'LeNetTrainLoader')
torch.save(le_val_loader, "LeNetValLoader")

# load data for AlexNet
alex_train_loader, alex_val_loader = split_dataloaders(G_TRAIN_IMG_PATH, train_size = 0.8, batch_size = G_BATCH_SIZE, num_workers = G_NUM_WORKERS, transform = train_transform_alex)
torch.save(alex_train_loader, 'AlexNetTrainLoader')
torch.save(alex_val_loader, "AlexNetValLoader")
