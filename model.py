import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Train and evaluate Fake Image Detection Model')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
parser.add_argument('--test_dir', type=str, required=True, help='Path to testing dataset')
parser.add_argument('--output_model', type=str, default='fake_image_model', help='Path to save trained model')
parser.add_argument('--output_results', type=str, default='test_results_300_lr.txt', help='Path to save test results')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

class FakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, sub_dir in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, sub_dir)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error loading image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Training Dataset
dataset = FakeImageDataset(args.train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Loaded {len(dataset)} training images")

# Load Training Dataset
dataset = FakeImageDataset(args.test_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"Loaded {len(dataset)} testimages")

class FakeImageModel(nn.Module):
    def __init__(self):
        super(FakeImageModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize Model
model = FakeImageModel().to(device)
print("Model Loaded")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_var, accuracy_var, test_loss_var, test_accuracy_var = [], [], [], []
best_accuracy = 0.0
best_loss=0.0

for epoch in range(args.epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    loss_var.append(train_loss)
    accuracy_var.append(train_accuracy)

    model.eval()
    test_running_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_loss_var.append(test_loss)
    test_accuracy_var.append(test_accuracy)
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_loss=test_loss
        torch.save(model.state_dict(), f"{args.output_model}_epoch_{epoch}_lr_{args.learning_rate}_ac_{best_accuracy}.pth") # saving the file name as the epochs with 
        print(f"New best model saved with accuracy: {best_accuracy:.2%}")
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2%}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), loss_var, label='Train Loss')
plt.plot(range(1, args.epochs+1), test_loss_var, label='Test Loss')
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('accuray_n_loss_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

# Save test results
with open(args.output_results, 'w') as f:
    # f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Best Test Accuracy: {best_accuracy:.2%}\n")
    f.write(f"Best Test loss : {best_loss:.4f}\n")

print(f"Test results saved to {args.output_results}")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Train and evaluate Fake Image Detection Model')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
parser.add_argument('--test_dir', type=str, required=True, help='Path to testing dataset')
parser.add_argument('--output_model', type=str, default='fake_image_model', help='Path to save trained model')
parser.add_argument('--output_results', type=str, default='test_results_300_lr.txt', help='Path to save test results')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

class FakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, sub_dir in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, sub_dir)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error loading image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Training Dataset
dataset = FakeImageDataset(args.train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Loaded {len(dataset)} training images")

# Load Training Dataset
dataset = FakeImageDataset(args.test_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"Loaded {len(dataset)} testimages")

class FakeImageModel(nn.Module):
    def __init__(self):
        super(FakeImageModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize Model
model = FakeImageModel().to(device)
print("Model Loaded")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_var, accuracy_var, test_loss_var, test_accuracy_var = [], [], [], []
best_accuracy = 0.0
best_loss=0.0

for epoch in range(args.epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    loss_var.append(train_loss)
    accuracy_var.append(train_accuracy)

    model.eval()
    test_running_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_loss_var.append(test_loss)
    test_accuracy_var.append(test_accuracy)
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_loss=test_loss
        torch.save(model.state_dict(), f"{args.output_model}_epoch_{epoch}_lr_{args.learning_rate}_ac_{best_accuracy}.pth") # saving the file name as the epochs with 
        print(f"New best model saved with accuracy: {best_accuracy:.2%}")
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2%}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), loss_var, label='Train Loss')
plt.plot(range(1, args.epochs+1), test_loss_var, label='Test Loss')
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('accuray_n_loss_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

# Save test results
with open(args.output_results, 'w') as f: 
    # f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Best Test Accuracy: {best_accuracy:.2%}\n")
    f.write(f"Best Test loss : {best_loss:.4f}\n")

print(f"Test results saved to {args.output_results}")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Train and evaluate Fake Image Detection Model')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
parser.add_argument('--test_dir', type=str, required=True, help='Path to testing dataset')
parser.add_argument('--output_model', type=str, default='fake_image_model', help='Path to save trained model')
parser.add_argument('--output_results', type=str, default='test_results_300_lr.txt', help='Path to save test results')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

class FakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, sub_dir in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, sub_dir)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error loading image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Training Dataset
dataset = FakeImageDataset(args.train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Loaded {len(dataset)} training images")

# Load Training Dataset
dataset = FakeImageDataset(args.test_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"Loaded {len(dataset)} testimages")

class FakeImageModel(nn.Module):
    def __init__(self):
        super(FakeImageModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize Model
model = FakeImageModel().to(device)
print("Model Loaded")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_var, accuracy_var, test_loss_var, test_accuracy_var = [], [], [], []
best_accuracy = 0.0
best_loss=0.0

for epoch in range(args.epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    loss_var.append(train_loss)
    accuracy_var.append(train_accuracy)

    model.eval()
    test_running_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_loss_var.append(test_loss)
    test_accuracy_var.append(test_accuracy)
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_loss=test_loss
        torch.save(model.state_dict(), f"{args.output_model}_epoch_{epoch}_lr_{args.learning_rate}_ac_{best_accuracy}.pth") # saving the file name as the epochs with 
        print(f"New best model saved with accuracy: {best_accuracy:.2%}")
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2%}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), loss_var, label='Train Loss')
plt.plot(range(1, args.epochs+1), test_loss_var, label='Test Loss')
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('accuray_n_loss_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

# Save test results
with open(args.output_results, 'w') as f:
    # f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Best Test Accuracy: {best_accuracy:.2%}\n")
    f.write(f"Best Test loss : {best_loss:.4f}\n")

print(f"Test results saved to {args.output_results}")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Train and evaluate Fake Image Detection Model')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
parser.add_argument('--test_dir', type=str, required=True, help='Path to testing dataset')
parser.add_argument('--output_model', type=str, default='fake_image_model', help='Path to save trained model')
parser.add_argument('--output_results', type=str, default='test_results_300_lr.txt', help='Path to save test results')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
args = parser.parse_args()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

class FakeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, sub_dir in enumerate(["real", "fake"]):
            folder = os.path.join(root_dir, sub_dir)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error loading image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.long)

# Image Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Training Dataset
dataset = FakeImageDataset(args.train_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Loaded {len(dataset)} training images")

# Load Training Dataset
dataset = FakeImageDataset(args.test_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

print(f"Loaded {len(dataset)} testimages")

class FakeImageModel(nn.Module):
    def __init__(self):
        super(FakeImageModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize Model
model = FakeImageModel().to(device)
print("Model Loaded")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
loss_var, accuracy_var, test_loss_var, test_accuracy_var = [], [], [], []
best_accuracy = 0.0
best_loss=0.0

for epoch in range(args.epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    loss_var.append(train_loss)
    accuracy_var.append(train_accuracy)

    model.eval()
    test_running_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    
    test_loss = test_running_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    test_loss_var.append(test_loss)
    test_accuracy_var.append(test_accuracy)
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_loss=test_loss
        torch.save(model.state_dict(), f"{args.output_model}_epoch_{epoch}_lr_{args.learning_rate}_ac_{best_accuracy}.pth") # saving the file name as the epochs with 
        print(f"New best model saved with accuracy: {best_accuracy:.2%}")
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2%}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), loss_var, label='Train Loss')
plt.plot(range(1, args.epochs+1), test_loss_var, label='Test Loss')
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('accuray_n_loss_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs+1), accuracy_var, label='Train Accuracy')
plt.plot(range(1, args.epochs+1), test_accuracy_var, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

# Save test results
with open(args.output_results, 'w') as f: 
    # f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Best Test Accuracy: {best_accuracy:.2%}\n")
    f.write(f"Best Test loss : {best_loss:.4f}\n")

print(f"Test results saved to {args.output_results}")
