import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import pandas as pd


class StanfordDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform

        strip_root = len(img_dir) + 1
        df = pd.DataFrame({'paths': [
            os.path.join(root[strip_root:], name)
            for root, _, files in os.walk(img_dir)
            for name in files
        ]})
        df[['xmin', 'ymin', 'xmax', 'ymax']] = df.apply(lambda x: self.bnd(x['paths']), axis=1, result_type='expand')

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

    def bnd(self, img_path):
        annot_path = os.path.join(self.annot_dir, img_path.replace('.jpg', ''))
        root = ET.parse(annot_path).getroot()

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        xmin = float(root.find('object/bndbox/xmin').text) / width
        ymin = float(root.find('object/bndbox/ymin').text) / height
        xmax = float(root.find('object/bndbox/xmax').text) / width
        ymax = float(root.find('object/bndbox/ymax').text) / height

        return xmin, ymin, xmax, ymax
