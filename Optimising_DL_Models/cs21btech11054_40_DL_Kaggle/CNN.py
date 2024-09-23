import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

train_dir = '/kaggle/input/iith-dl-contest-2024/train/train'
train_data  = datasets.ImageFolder(train_dir, transform = transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Training the model
model = conv_sa().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(10)):  # Number of epochs
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load test data
test_dir = '/kaggle/input/iith-dl-contest-2024/test'
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

# Generate predictions
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.append(train_data.classes[predicted.item()])

# Save predictions to .csv file
submission = pd.DataFrame({'ID': [os.path.basename(path) for path, _ in test_data.imgs], 'Category': predictions})
submission.to_csv('submission.csv', index=False)