import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Configuration
BATCH_SIZE = 64
EPOCHS = 20
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

# Define the model
class conv_sa(nn.Module):
    def __init__(self):
        super(conv_sa, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 7)  
        self.Relu = nn.ReLU()        
        self.conv2 = nn.Conv2d(32, 64, 7)       
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 5)               
        self.conv4 = nn.Conv2d(128, 256, 5)        
        self.gap = nn.AvgPool2d(7,7)
        
        self.l1 = nn.Linear(256, 192)
        self.l2 = nn.Linear(192, 50)
        
    def forward(self, x):
        out = self.Relu(self.conv1(x))
        out = self.Relu(self.conv2(out))       
        out = self.pool(out)
        out = self.Relu(self.conv3(out))
        out = self.pool(out)       
        out = self.Relu(self.conv4(out))
       
        out = self.gap(out)
        out = out.view((x.shape[0], 256))
        
        out = self.Relu(self.l1(out))
        out = self.l2(out)
        return out

# Define the model
model = conv_sa().to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters())

# Loss function
criterion = nn.CrossEntropyLoss()


# Data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Datasets
train_dataset = ImageDataset('/kaggle/input/iith-dl-contest-2024/train/train', transform=transform)
test_dataset = ImageDataset('/kaggle/input/iith-dl-contest-2024/test/test', transform=transform, train=False)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Training function
def train(model):
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
def test(model):
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
best_model = conv_sa().to(DEVICE)
for epoch in range(EPOCHS):
    train_loss = train(best_model)
    print(f'Epoch: {epoch+1}, Loss: {train_loss}')
    if train_loss < min_loss:
        min_loss = train_loss
        best_model = model

# Make predictions
best_model.eval()
predictions = test(best_model)

# Save predictions to a .csv file
df = pd.DataFrame(predictions, columns=['ID', 'Category'])
df.to_csv('submission.csv', index=False)
