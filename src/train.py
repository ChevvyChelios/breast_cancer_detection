
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from src.dataset import BreastUltrasoundDataset
from src.model import create_model


def compute_class_weights(train_csv_path):
    df = pd.read_csv(train_csv_path)
    counts = df["label"].value_counts().sort_index()
    total = counts.sum()
    # inverse frequency: more weight for rare classes
    weights = total / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, all_labels, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_csv = os.path.join("metadata", "train2.csv")
    val_csv = os.path.join("metadata", "val2.csv")

    batch_size = 16
    num_epochs = 25
    lr = 1e-4

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []


    # Datasets & loaders
    train_dataset = BreastUltrasoundDataset(train_csv, augment=True)
    val_dataset = BreastUltrasoundDataset(val_csv, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = create_model(num_classes=3, pretrained=True)
    model = model.to(device)

    # Loss with class weights
    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = os.path.join("models", "model2.pt")
    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = eval_model(model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{num_epochs} "
              f"- {elapsed:.1f}s | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Print small report every few epochs
        if (epoch + 1) % 5 == 0:
            print("Validation classification report:")
            print(classification_report(y_true, y_pred))

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved best model with val acc = {best_val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)


    print("Training finished. Best val acc:", best_val_acc)

    with open("models/training_metrics.json", "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_acc": train_accs,
            "val_acc": val_accs
        }, f, indent=2)

    # Final detailed report
    print("Final validation confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

# def compute_class_weights(train_csv_path):
#     df = pd.read_csv(train_csv_path)
#     counts = df["label"].value_counts().sort_index()
#     total = counts.sum()
#     # inverse frequency: more weight for rare classes
#     weights = total / (len(counts) * counts)
#     return torch.tensor(weights.values, dtype=torch.float)


# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for imgs, labels in loader:
#         imgs = imgs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * imgs.size(0)
#         _, preds = torch.max(outputs, 1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#     return running_loss / total, correct / total


# def eval_model(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * imgs.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#             all_labels.extend(labels.cpu().tolist())
#             all_preds.extend(preds.cpu().tolist())

#     avg_loss = running_loss / total
#     acc = correct / total
#     return avg_loss, acc, all_labels, all_preds


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     train_csv = os.path.join("metadata", "train.csv")
#     val_csv = os.path.join("metadata", "val.csv")

#     batch_size = 16
#     num_epochs = 25
#     lr = 1e-4

#     # Datasets & loaders
#     train_dataset = BreastUltrasoundDataset(train_csv, augment=True)
#     val_dataset = BreastUltrasoundDataset(val_csv, augment=False)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#     # Model
#     model = create_model(num_classes=3, pretrained=True)
#     model = model.to(device)

#     # Loss with class weights
#     class_weights = compute_class_weights(train_csv).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     best_val_acc = 0.0
#     best_model_path = os.path.join("models", "model1.pt")
#     os.makedirs("models", exist_ok=True)

#     for epoch in range(num_epochs):
#         start = time.time()
#         train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_loss, val_acc, y_true, y_pred = eval_model(model, val_loader, criterion, device)

#         elapsed = time.time() - start
#         print(f"Epoch {epoch+1}/{num_epochs} "
#               f"- {elapsed:.1f}s | "
#               f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
#               f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

#         # Print small report every few epochs
#         if (epoch + 1) % 5 == 0:
#             print("Validation classification report:")
#             print(classification_report(y_true, y_pred))

#         # Save best
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), best_model_path)
#             print(f"--> Saved best model with val acc = {best_val_acc:.4f}")

#     print("Training finished. Best val acc:", best_val_acc)

#     # Final detailed report
#     print("Final validation confusion matrix:")
#     print(confusion_matrix(y_true, y_pred))


# if __name__ == "__main__":
#     main()
