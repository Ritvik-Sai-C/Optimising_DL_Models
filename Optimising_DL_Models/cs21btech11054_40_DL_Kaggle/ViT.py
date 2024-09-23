import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
from tqdm import tqdm
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer

# Define the ViT model
class ViTModel(nn.Module):
    def _init_(self, num_classes):
        super(ViTModel, self)._init_()
        self.model = VisionTransformer(img_size=224, patch_size=16, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Data preprocessing and augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = ImageFolder(root='/kaggle/input/iith-dl-contest-2024/train', transform=train_transform)
test_dataset = ImageFolder(root='/kaggle/input/iith-dl-contest-2024/test', transform=test_transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
model = ViTModel(num_classes=len(train_dataset.classes))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1  # Increase the number of epochs for better performance
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in tqdm_bar:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        tqdm_bar.set_postfix(loss=running_loss / ((len(tqdm_bar) - 1) * train_loader.batch_size))
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), 'trained_model.pth')

# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())

# Generate submission file
submission_df = pd.DataFrame({'ID': test_dataset.samples, 'Category': predictions})
submission_df['ID'] = submission_df['ID'].apply(lambda x: x.split('/')[-1])
submission_df.to_csv('submission.csv', index=False)