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
    
    def __init__(self, input_features):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        # self.double()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x