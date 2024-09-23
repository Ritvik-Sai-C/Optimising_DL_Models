import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Configuration
BATCH_SIZE = 64
EPOCHS = 10
MODEL_PATH = 'best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.classes = os.listdir(root_dir)
            self.paths = []
            self.labels = []
            for i, cls in enumerate(self.classes):
                for img in os.listdir(os.path.join(root_dir, cls)):
                    self.paths.append(os.path.join(root_dir, cls, img))
                    self.labels.append(i)
        else:
            self.paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')  # Convert image to RGB
        if self.transform:
            img = self.transform(img)
        if self.train:
            return img, self.labels[idx]
        else:
            return img, self.paths[idx]

# Model
model = resnet50(pretrained=False).to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Loss function
criterion = nn.CrossEntropyLoss()

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets
train_dataset = ImageDataset('/kaggle/input/iith-dl-contest-2024/train/train', transform=transform)
test_dataset = ImageDataset('/kaggle/input/iith-dl-contest-2024/test/test', transform=transform, train=False)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Training function
def train():
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test():
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, paths in tqdm(test_loader):  # Ignore the second return value
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                predictions.append((paths[i], train_dataset.classes[predicted[i]]))
    return predictions

# Training loop
min_loss = float('inf')
best_model = None
for epoch in range(EPOCHS):
    train_loss = train()
    print(f'Epoch: {epoch+1}, Loss: {train_loss}')
    if train_loss < min_loss:
        min_loss = train_loss
        best_model = model

# Make predictions
best_model.eval()
predictions = []
with torch.no_grad():
    for images, paths in tqdm(test_loader):  # Ignore the second return value
        images = images.to(DEVICE)
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            predictions.append((paths[i], train_dataset.classes[predicted[i]]))

# Save predictions to a .csv file
df = pd.DataFrame(predictions, columns=['ID', 'Category'])
df.to_csv('submission.csv', index=False)