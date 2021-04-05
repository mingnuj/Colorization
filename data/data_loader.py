from PIL import Image
import cv2
import os
import torch.utils.data as data
from torch.utils.data import DataLoader
from data.transform import CustomTransform


class ImageLoader(data.Dataset):
    def __init__(self, file_path, size=256, mode="train", hintpercent=2):
        self.file_path = os.path.join(file_path, mode)
        self.hint_percent = hintpercent
        self.transforms = CustomTransform(size, hintpercent)
        self.files = []
        for image in os.listdir(self.file_path):
            self.files.append(os.path.join(self.file_path, image))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        l, gt, ab = self.transforms(img)
        return l, gt, ab
