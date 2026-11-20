import torch
import torch.nn as nn


# Metrics

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return (preds == targets).sum().item(), targets.size(0)

def f1_score(outputs, targets, num_classes):
    _, preds = torch.max(outputs, dim=1)

    f1_total = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_total += f1

    return f1_total / num_classes


# Training

def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_f1 = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        correct, n = accuracy(outputs, targets)
        total_correct += correct
        total_samples += n

        total_f1 += f1_score(outputs, targets, num_classes) * inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_f1 = total_f1 / total_samples

    return avg_loss, avg_acc, avg_f1


def train(model, train_loader, epochs, lr, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    num_classes = len(train_loader.dataset.classes)

    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")
