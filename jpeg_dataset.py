from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ImageInfo:
    img_path: str
    label: int


class JpegDataset(Dataset):
    def __init__(self, images_info: List[ImageInfo], transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_data = self.__read_images(images_info)

    def __len__(self):
        return len(self.img_data)

    def __read_images(self, images_info: List[ImageInfo]):
        img_data = []
        for img_info in images_info:
            img_data.append((img_info.img_path, img_info.label))
        return img_data

    def __getitem__(self, index):
        img_name, label = self.img_data[index]
        with Image.open(img_name).convert('RGB') as sample_image:
            image = np.asarray(sample_image)
        if self.transform is not None:
            image = self.transform(image)
        return label, img_name, image
