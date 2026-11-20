import torch
import torch.nn as nn

def init_normal(model, mean=0.0, std=0.01):
    """
    Apply normal (Gaussian) initialization to all linear and conv layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=mean, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
