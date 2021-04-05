from torchvision import transforms
import cv2
import numpy as np


class CustomTransform(object):
    def __init__(self, size, percent=2):
        self.size = size
        self.percent = percent
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def hint_img_ab(self, ab, percent=2):
        hint_img = np.zeros_like(ab)
        nonzero = 0
        while nonzero <= percent:
            height = np.random.randint(low=0, high=ab.shape[0])
            width = np.random.randint(low=0, high=ab.shape[1])
            hint_img[height, width, :] = ab[height, width, :]
            nonzero = (np.count_nonzero(hint_img) / (ab.shape[0] * ab.shape[1] * ab.shape[2])) * 100
        return hint_img

    def __call__(self, img):
        image = cv2.resize(img, (self.size, self.size))
        l, ab = self.bgr_to_lab(image)
        hint_mask = self.hint_img_ab(ab, self.percent)
        # hint_image = np.concatenate((l[:, :, np.newaxis], hint_mask), axis=-1)

        return self.transform(l), self.transform(ab), self.transform(hint_mask)
