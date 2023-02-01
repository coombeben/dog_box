import os
import xml.etree.ElementTree as ET
import pandas as pd

from consts import img_dir, annot_dir


def bnd(img_path):
    annot_path = os.path.join(annot_dir, img_path.replace('.jpg', ''))
    root = ET.parse(annot_path).getroot()

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    xmin = float(root.find('object/bndbox/xmin').text) / width
    ymin = float(root.find('object/bndbox/ymin').text) / height
    xmax = float(root.find('object/bndbox/xmax').text) / width
    ymax = float(root.find('object/bndbox/ymax').text) / height

    return xmin, ymin, xmax, ymax


strip_root = len(img_dir) + 1
df = pd.DataFrame({'paths': [
    os.path.join(root[strip_root:], name)
    for root, _, files in os.walk(img_dir)
    for name in files
]})
df[['xmin', 'ymin', 'xmax', 'ymax']] = df.apply(lambda x: bnd(x['paths']), axis=1, result_type='expand')

df.to_csv('data.csv', index=False)
