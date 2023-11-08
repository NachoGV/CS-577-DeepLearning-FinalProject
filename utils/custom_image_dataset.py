import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = os.listdir(img_dir)  # Assuming each directory is a different class
        self.img_paths = []
        self.class_count = {}
        for c in self.classes:
            class_dir = os.path.join(img_dir, c)
            self.img_paths += [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
            self.class_count[c] = len(os.listdir(class_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR))
        label = self.classes.index(os.path.basename(os.path.dirname(img_path)))  # Get class index
        if self.transform:
            image = self.transform(image)
        return image, label