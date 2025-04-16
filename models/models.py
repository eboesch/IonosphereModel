from torch import nn
import torch.nn.functional as F
from typing import Any


def get_model_class_from_string(model_str: str) -> Any:
    if model_str == "FCN":
        return FCN
    
    else:
        assert False, f"model class {model_str} is not supported."


class FCN(nn.Module):
    """
    Class for a fully connected model in pytorch.
    """
    
    def __init__(self, input_features, num_hidden_layers, hidden_size):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_features, hidden_size))
        self.layers.append(nn.ReLU())

        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size, 1))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x