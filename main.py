import torch
import torch.nn as nn

from data.mnist import get_mnist_loaders
from data.cifar10 import get_cifar10_loaders

from models.mlp import MLP
from models.cnn import CNN

from init import init_xavier, init_kaiming, init_normal, init_uniform

from train import train
from test import evaluate_test


def main():
    # ----------------------------
    # Device
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Experiment config
    # ----------------------------
    dataset_name = "cifar10"      # "mnist" or "cifar10"
    model_name = "mlp"          # "mlp" or "cnn"

    batch_size = 128
    epochs = 10
    lr = 0.01

    activation = nn.ReLU
    init_fn = init_uniform


    print("===================================")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Model:   {model_name.upper()}")
    print(f"Epochs:  {epochs}")
    print(f"Weight initialization: {init_fn.__name__}")
    print("===================================")


    # ----------------------------
    # Dataset
    # ----------------------------
    if dataset_name == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size)
        input_channels = 1
        input_height = 28
        input_width = 28
        num_classes = 10
        input_dim = 28 * 28

    elif dataset_name == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size)
        input_channels = 3
        input_height = 32
        input_width = 32
        num_classes = 10
        input_dim = 32 * 32 * 3

    else:
        raise ValueError("Unknown dataset")

    # ----------------------------
    # Model
    # ----------------------------
    if model_name == "mlp":
        model = MLP(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=num_classes,
            depth=3,
            activation=activation,
            init_fn=init_fn
        )

    elif model_name == "cnn":
        model = CNN(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            conv_configs=[(32, 3), (64, 3)],
            fc_hidden=256,
            num_classes=num_classes,
            activation=activation,
            init_func=init_fn,
            dropout=0.5
        )

    else:
        raise ValueError("Unknown model")

    # ----------------------------
    # Train
    # ----------------------------
    train(
        
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        device=device
    )

    # ----------------------------
    # Test
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    evaluate_test(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
