import torch.nn as nn
import torchvision.models as models


def load_model(device):
    """Loads the modified vgg16 model to device"""
    model = models.vgg16(weights='IMAGENET1K_FEATURES')
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 64),
        nn.Sigmoid(),
        nn.Linear(64, 4)
    ).to(device)

    return model
