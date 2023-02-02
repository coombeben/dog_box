import os
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import pandas as pd


class StanfordDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv('data.csv')
        df['paths'] = df['paths'].str.replace('\\', os.path.sep, regex=False)

        self.data = df

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.data['paths'][item])

        image = io.read_image(img_path, io.ImageReadMode.RGB)
        bndbox = torch.tensor(self.data.iloc[item, 1:], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, bndbox
