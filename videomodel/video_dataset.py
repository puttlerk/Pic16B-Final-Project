import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def idx_to_class(directory):
    labels = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not labels:
        raise FileNotFoundError(f"Couldn't find any label folder n {directory}.")
    return {i: label for i, label in enumerate(labels)}

class VideoDataset(Dataset):

    def __init__(self, ids, idx_to_label, transform):
        self.ids = ids
        self.idx_to_label = idx_to_label
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem(self, index):
        frame_paths = glob.glob(self.ids[index] + "/*.jpg")
        label = self.idx_to_label[index]
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            frame = self.transform(frame)
            if frame.is_tensor():
                frames.append(frame)
            else:
                frame = transforms.ToTensor(frame)
            frames.append(frame)
        frames = frames.stack()
        return frames, label


# transform = transforms.Compose([
#     transforms.Resize((32,32)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# 
# vtrainpath = "../asl_video_frames/train"
# vtrainids = glob.glob(vtrainpath + "/*/*")
# vtrainlabels = sorted(entry.name for entry in os.scandir(vtrainpath) if entry.is_dir())
# if not vtrainlabels:
#     raise FileNotFoundError(f"Couldn't find any class folder in {vtrainpath}.")
# vtrain_idx_to_class = {i: label for i, label in enumerate(vtrainlabels)}
# vtraindata = VideoDataset(vtrainids, vtrain_idx_to_class, transform)
# vtrain_loader = torch.utils.data.DataLoader(vtraindata, batch_size = 64, shuffle = True)