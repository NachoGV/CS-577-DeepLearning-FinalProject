import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from image_enhancement_functions import histogram_equalization, clahe, color_balance_adjustment, min_max_contrast_enhancement

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, box_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = os.listdir(img_dir)  # Assuming each directory is a different class
        self.img_paths = []
        self.box_paths = []
        self.class_count = {}
        for c in self.classes:

            # Image
            class_dir = os.path.join(img_dir, c)
            self.img_paths += [os.path.join(class_dir, img) for img in os.listdir(class_dir)]

            # Bounding box
            class_dir = os.path.join(box_dir, c)
            self.box_paths += [os.path.join(class_dir, box) for box in os.listdir(class_dir)]

            # Class count
            self.class_count[c] = len(os.listdir(class_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # Image
        img_path = self.img_paths[idx]
        image = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR))
        if self.transform:
            image = self.transform(image)

        # Label
        label = self.classes.index(os.path.basename(os.path.dirname(img_path)))  # Get class index

        # Bounding box
        box_path = self.box_paths[idx]
        box = []
        with open(box_path, 'r') as f:
            line = f.readlines()[1]
            box = line.split(' ')[1:5]

        return image, label, box
    
# ----------------------------------------------------------------------------------------------------------

class CocoDetection(CocoDetection):
    def __init__(
        self, 
        annotation_file_path: str,
        image_directory_path: str, 
        image_processor, 
        train: bool = True,
    ):
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target