import torch
import torch.nn as nn

def init_xavier(model):
    """
    Apply Xavier/Glorot initialization to all linear and conv layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
