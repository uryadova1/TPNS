import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
from PIL import Image

import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 28x28 -> 24x24
        self.pool = nn.AvgPool2d(2, 2)  # 24x24 -> 12x12
        self.conv2 = nn.Conv2d(6, 16, 5)  # 12x12 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def plot_confusion_matrix(model):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")


# def train_model(model):
#     X = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
#     y = torch.tensor(train_labels, dtype=torch.long)
#     dataset = TensorDataset(X, y)
#     loader = DataLoader(dataset, batch_size=64, shuffle=True)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(5):
#         for batch_X, batch_y in loader:
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def train_model(model):
    X = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(train_labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 6
    losses = list()
    accuracies = list()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        losses.append(avg_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, epochs + 1), losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper right')
    ax1.set_title("Training Loss")

    color = 'tab:blue'
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(range(1, epochs + 1), [acc * 100 for acc in accuracies], color=color, label='Training Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper left')
    ax2.set_title("Training Accuracy")

    plt.tight_layout()
    plt.show()


def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


train_images = load_images('train-images-idx3-ubyte.gz')

train_labels = load_labels('train-labels-idx1-ubyte.gz')

model = LeNet5()

train_model(model)

test_images = load_images('t10k-images-idx3-ubyte.gz')
test_labels = load_labels('t10k-labels-idx1-ubyte.gz')

X_test = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
evaluate(model, test_loader)

plot_confusion_matrix(model)
