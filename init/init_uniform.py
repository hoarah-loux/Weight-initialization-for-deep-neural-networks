import torch
import torch.nn as nn

def init_uniform(model, a=0.1):
    """
    Apply uniform random initialization to all linear and conv layers.
    Range is [-a, a].
    """
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.uniform_(m.weight, -a, a)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
