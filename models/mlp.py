import torch
import torch.nn as nn
from init import init_xavier,init_kaiming,init_normal,init_uniform


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, depth, activation, init_fn: callable):
        super().__init__()
        
        

        
        self.layers = nn.ModuleList()

        
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        
        self.activation = activation()

        
        self.output = nn.Linear(hidden_dim, num_classes)

        init_fn(self)

    def forward(self, x):
        
        x = x.view(x.size(0), -1)

        
        for layer in self.layers:
            x = self.activation(layer(x))

        
        x = self.output(x)
        return x
