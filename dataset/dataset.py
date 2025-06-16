from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
# from segment_anything import sam_model_registry, SamPredictor  # 移除SAM相关
import os
import matplotlib.pyplot as plt
from time import sleep
import random


def load_data(images_path, labels_path=None):
    images = np.load(images_path)
    if labels_path is not None:
        labels = np.load(labels_path)
        return images, labels
    return images, None

def sigmoid_contrast(img, gain=75, cutoff=0.90):
    img = img.astype(np.float32) / 255.0
    img = 1 / (1 + np.exp(-gain * (img - cutoff)))
    img = (img * 255).clip(0, 255).astype(np.uint8)

    img = np.mean(img, axis=2, keepdims=True).repeat(3, axis=2)

    return img

def filter_data(data):
    # if isinstance(data, np.ndarray) and data.ndim == 3:
    #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    #     axs[0].imshow(data)
    #     axs[0].set_title('Original')
    #     axs[0].axis('off')
    #     filtered = sigmoid_contrast(data)
    #     axs[1].imshow(filtered)
    #     axs[1].set_title('Filtered')
    #     axs[1].axis('off')
    #     plt.show()
    #     sleep(1000000)
    data = data.mean(axis=2, keepdims=True).repeat(3, axis=2) 

    if isinstance(data, np.ndarray):
        return sigmoid_contrast(data)
    else:
        return np.array([sigmoid_contrast(img) for img in data])


class MyDataset(Dataset):
    def __init__(self, cfg, images_path, labels_path=None):
        self.cfg = cfg
        self.images_path = images_path
        self.data, self.labels = load_data(images_path, labels_path)
        if self.labels is not None:
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if "train_images" in self.images_path and self.labels[idx] != 6 and self.labels[idx] != 9:
                if random.random() < 1/3:
                    angle = random.choice([90, 180, 270])
                    img = np.rot90(img, k=angle // 90)




        if self.cfg['data']['filter']:
            imgf = filter_data(img)

        img = imgf + img  # 使用滤波后的图像

        img = img.astype(np.uint8)  # 确保是 uint8 类型

        img = Image.fromarray(img)
        if self.cfg['model']['type'] == 'cnn':
            img = img.resize((32, 32))
        else:
            img = img.resize((224, 224))

        img = np.array(img)  

        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1).float() 


        # plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        # plt.title(f"Index: {idx}")
        # plt.axis('off')
        # plt.show()

        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img, 0
