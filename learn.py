import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Resize
from tqdm import tqdm

from data_prep import StanfordDataset
from model import load_model
from consts import *

# Define dataset/loader
transforms = Compose([
    Resize((224, 224)),
    ConvertImageDtype(torch.float32),
    # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

stanford_dataset = StanfordDataset(img_dir, transforms)
train_data = DataLoader(dataset=stanford_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


# Define model
model = load_model(device)

loss_function = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Train model
for epoch in range(NUM_EPOCHS):
    print(f'Epoch: {epoch}')
    losses = []
    for imgs, bndboxs in tqdm(train_data):
        imgs = imgs.to(device)
        bndboxs = bndboxs.to(device)

        optimiser.zero_grad()
        pred = model(imgs)
        loss = loss_function(pred, bndboxs)
        loss.backward()

        optimiser.step()

        losses.append(loss)
    print(f'Mean loss: {sum(losses)/len(losses):.5f}')
    print('-' * 50)

# Save model
torch.save(model.state_dict(), 'model.pth')
