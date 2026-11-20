import torch
import torch.nn as nn

def init_kaiming(model):
    """
    Apply He/Kaiming initialization to all linear and conv layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
