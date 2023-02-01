import torch
import matplotlib.pyplot as plt
import torchvision.utils as utils
from torchvision.transforms import Compose, ConvertImageDtype, Resize
import random

from data_prep import StanfordDataset
from model import load_model
from consts import model_path, device, img_dir, annot_dir


def draw_bndbox(img, bnd):
    img = img.squeeze()
    img, bnd = torch.clamp(img * 224, 0, 224).type(torch.uint8), torch.clamp(bnd * 224, 0, 224).type(torch.uint8)
    img = utils.draw_bounding_boxes(img, bnd, width=3, colors=(0, 255, 0))
    plt.imshow(img.permute(1, 2, 0))
    plt.show()


model = load_model(device)
model.load_state_dict(torch.load(model_path))
model.eval()

transforms = Compose([
    Resize((224, 224)),
    ConvertImageDtype(torch.float32),
])
stanford_dataset = StanfordDataset(img_dir, annot_dir, transforms)


for _ in range(10):
    idx = random.randint(0, len(stanford_dataset))
    img, bndbox = stanford_dataset[idx]
    img, bndbox = img.unsqueeze(0).to(device), bndbox.unsqueeze(0).to(device)

    pred = model(img)

    draw_bndbox(img, pred)
