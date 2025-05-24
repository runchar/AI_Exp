from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch

def load_data(images_path, labels_path=None):
    images = np.load(images_path)
    if labels_path is not None:
        labels = np.load(labels_path)
        return images, labels
    return images, None

def filter_data(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if sum(data[i, j, :]) <= 254*3:
                data[i, j, :] = [0, 0, 0] 
    return data


class MyDataset(Dataset):
    def __init__(self, cfg, images_path, labels_path=None):
        self.cfg = cfg
        self.data, self.labels = load_data(images_path, labels_path)
        if self.labels is not None:
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.cfg['data']['filter']:
            img = filter_data(img)

        img = Image.fromarray(img.astype('uint8'))
        img = img.resize((224, 224))
        img = np.array(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (H,W,C) -> (C,H,W)

        return img, self.labels[idx]

