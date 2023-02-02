import os
import platform
import torch

# Learning parameters
TRAIN_SPLIT = 0.8
LEARNING_RATE = 10 ** -4
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Folder locations
root = os.path.dirname(__file__)
model_path = os.path.join(root, 'model.pth')

if platform.system() == 'Windows':
    archive_dir = r"C:\Users\coomb\Pictures\archive"
else:
    archive_dir = os.path.join(root, 'archive')

img_dir = os.path.join(archive_dir, 'images', 'Images')
annot_dir = os.path.join(archive_dir, 'annotations', 'Annotation')
