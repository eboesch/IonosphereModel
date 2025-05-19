from torch import nn
import torch.nn.functional as F
from typing import Any
import torch
import yaml

def get_model(config: dict, input_size: int) -> Any:
    model_type = config["model_type"]

    if model_type == "FCN":
        model = FCN(input_size, config["num_hidden_layers"], config["hidden_size"], 1)
    elif model_type == "TwoStageModel":
        # NOTE: older models didn't have the two-tier format of optional_features.
        # for the sake of backwards comptatibility we make this distinction here
        if type(config["optional_features"]) == dict:
            input_size = input_size - len(config["optional_features"]['delayed'])
            output_size_1 = config["output_size_1"] + len(config["optional_features"]['delayed'])
        else:
            input_size = input_size - len(config["optional_features"])
            output_size_1 = config["output_size_1"] + len(config["optional_features"])

        return TwoStageModel(
            input_size, 
            config["num_hidden_layers_1"],
            config["hidden_size_1"],
            config["output_size_1"],
            output_size_1,
            config["num_hidden_layers_2"],
            config["hidden_size_2"],
        )
    
    else:
        assert False, f"model class {model_type} is not supported."

    return model


def load_pretrained_model(pretrained_model_path: str):
    # NOTE: We are saving models with torch.save(model.state_dict), which makes the saved object a dictionary rather
    # than the full model class. For this reason, we have to reinstantiate the model class using the saved pretraining config.
    pretraining_config_path = pretrained_model_path + "trainig_config.yaml"
    with open(pretraining_config_path, 'r') as file:
        pretraining_config = yaml.load(file, Loader=yaml.FullLoader)

    model_state_dict = torch.load(pretrained_model_path + "model.pth", weights_only=False, map_location=torch.device('cpu'))
    input_size = model_state_dict[list(model_state_dict.keys())[0]].shape[1]
    if pretraining_config["model_type"] == "TwoStageModel":
        # NOTE: older models didn't have the two-tier format of optional_features.
        # for the sake of backwards comptatibility we make this distinction here
        if type(pretraining_config["optional_features"]) == dict:
            input_size += len(pretraining_config['optional_features']['delayed'])
        else:
            input_size += len(pretraining_config['optional_features'])

    model = get_model(pretraining_config, input_size)
    model.load_state_dict(model_state_dict)

    if pretraining_config["model_type"] == "TwoStageModel":
        for param in model.fcn1.parameters():
            param.requires_grad = False

        for param in model.fcn2.parameters():
            param.requires_grad = True
    
    elif pretraining_config["model_type"] == "FCN":
        for param in model.parameters():
            param.requires_grad = True

    return model


def load_model(model_path):
    model_config_path = model_path + "trainig_config.yaml"
    with open(model_config_path, 'r') as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)

    if model_config["pretrained_model_path"] is None:
        return load_pretrained_model(model_path)
    
    else:
        model = load_model(model_config["pretrained_model_path"])
        model_state_dict = torch.load(model_path + "model.pth", weights_only=False, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
        return model

class FCN(nn.Module):
    """
    Class for a fully connected model in pytorch.
    """
    
    def __init__(self, input_size, num_hidden_layers, hidden_size, output_size=1):
        super().__init__()

        self.layers = nn.ModuleList()
        if num_hidden_layers == 0:
            self.layers.append(nn.Linear(input_size, output_size))
        else:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.layers.append(nn.ReLU())

            for i in range(num_hidden_layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Linear(hidden_size, output_size))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TwoStageModel(nn.Module):
    def __init__(
        self,
        input_size_1,
        num_hidden_layers_1,
        hidden_size_1,
        output_size_1,
        input_size_2,
        num_hidden_layers_2,
        hidden_size_2,
    ):
        super().__init__()
        self.input_size_1 = input_size_1
        self.fcn1 = FCN(input_size_1, num_hidden_layers_1, hidden_size_1, output_size_1)
        self.fcn2 = FCN(input_size_2, num_hidden_layers_2, hidden_size_2, 1)
    
    def forward(self, x):
        x_1 = x[:, :self.input_size_1]
        x_2 = x[:, self.input_size_1:]
        h = self.fcn1(x_1)
        hx_2 = torch.cat([h, x_2], dim=1)
        out = self.fcn2(hx_2)
        return out
        

if __name__ == "__main__":
    x = torch.ones([10, 10])
    model = TwoStageModel(5, 3, 200, 200, 205, 3, 200, 1)
    model(x)