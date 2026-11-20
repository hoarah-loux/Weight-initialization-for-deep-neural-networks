import torch
import torch.nn as nn
from typing import Callable
from init import init_kaiming, init_normal, init_uniform, init_xavier

class CNN(nn.Module):
    def __init__(self, input_channels: int, input_height: int, input_width: int,
                 conv_configs: list, fc_hidden: int, num_classes: int,
                 activation: nn.Module, init_func: Callable, dropout: float = 0.5):
       
        super().__init__()

        

        if init_func is None:
            raise ValueError("You must provide a weight initialization function!")

        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(2, 2)

        # Build convolutional layers
        self.convs = nn.ModuleList()
        in_channels = input_channels
        for out_channels, k in conv_configs:
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2))
            in_channels = out_channels

        # Compute flattened size for FC layer after conv + pooling
        h, w = input_height, input_width
        for _ in conv_configs:
            h = h // 2  
            w = w // 2  
        fc_input_dim = h * w * in_channels

        
        self.fc = nn.Linear(fc_input_dim, fc_hidden)
        self.out = nn.Linear(fc_hidden, num_classes)

        
        init_func(self)

    def forward(self, x):
        
        for conv in self.convs:
            x = self.activation(conv(x))
            x = self.pool(x)

        
        x = torch.flatten(x, 1)

        
        x = self.activation(self.fc(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
