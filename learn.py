import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Resize, Normalize
from tqdm import tqdm

from data_prep import StanfordDataset
from model import load_model
from consts import *

# Define dataset/loader
transforms = Compose([
    Resize((224, 224)),
    ConvertImageDtype(torch.float32),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

stanford_dataset = StanfordDataset(img_dir, transforms)
train_size = int(len(stanford_dataset) * TRAIN_SPLIT)
test_size = len(stanford_dataset) - train_size
train, test = torch.utils.data.random_split(stanford_dataset, [train_size, test_size])
train_data = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_data = DataLoader(dataset=test, batch_size=BATCH_SIZE, pin_memory=True)

# Define model
model = load_model(device)

loss_function = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Train model
for epoch in range(NUM_EPOCHS):
    print(f'Epoch: {epoch}')
    train_losses = []
    model.train()

    for imgs, bndboxs in tqdm(train_data):
        imgs = imgs.to(device)
        bndboxs = bndboxs.to(device)

        optimiser.zero_grad()
        pred = model(imgs)
        loss = loss_function(pred, bndboxs)
        loss.backward()

        optimiser.step()

        train_losses.append(loss)
        
    test_losses = []
    model.eval()
    for test_img, test_bndbox in test_data:
        test_img, test_bndbox = test_img.to(device), test_bndbox.to(device)

        pred = model(test_img)
        loss = loss_function(pred, test_bndbox)

        test_losses.append(loss)

    print(f'Train loss: {sum(train_losses)/len(train_losses):.5f} | Test loss: {sum(test_losses)/len(test_losses):.5f}')
    print('-' * 50)

# Save model
torch.save(model.state_dict(), 'model.pth')
