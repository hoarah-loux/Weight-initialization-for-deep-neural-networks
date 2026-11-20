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



# Evaluation

def evaluate_test(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_f1 = 0.0

    num_classes = len(test_loader.dataset.classes)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            correct, n = accuracy(outputs, targets)
            total_correct += correct
            total_samples += n

            total_f1 += f1_score(outputs, targets, num_classes) * inputs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    avg_f1 = total_f1 / total_samples

    print(
        f"Test Loss: {avg_loss:.4f} | "
        f"Test Acc: {avg_acc:.4f} | "
        f"Test F1: {avg_f1:.4f}"
    )

    return avg_loss, avg_acc, avg_f1
