"""summary."""

import pathlib

from kan import KANLinear
from torch import nn
from utils import load_yaml


class Model(nn.Module):
    """summary."""

    def __init__(self, sequential_layer: nn.Sequential):
        super(Model, self).__init__()
        self.sequential_layer = sequential_layer

    def forward(self, x):
        x = self.sequential_layer(x)

        return x

    @classmethod
    def from_dict(cls, n_observations: int, n_actions: int, config_dict: dict):
        layers = []
        in_features = n_observations

        for i, layer in enumerate(config_dict["layers"]):
            layer_args = layer.split(" ")

            match layer_args[0]:
                case "linear":
                    if layer_args[1] != "final":
                        layer_args[1] = int(layer_args[1])
                        layers.append(
                            nn.Linear(
                                in_features=in_features, out_features=layer_args[1]
                            )
                        )
                    else:
                        layers.append(
                            nn.Linear(in_features=in_features, out_features=n_actions)
                        )
                        break
                    in_features = layer_args[1]

                case "kan_linear":
                    if layer_args[1] != "final":
                        layer_args[1] = int(layer_args[1])
                        layers.append(
                            KANLinear(
                                in_features=in_features, out_features=layer_args[1]
                            )
                        )
                    else:
                        layers.append(
                            KANLinear(in_features=in_features, out_features=n_actions)
                        )
                        break
                    in_features = layer_args[1]

                case "relu":
                    layers.append(nn.ReLU())

        sequential_layer = nn.Sequential(*layers)

        return cls(sequential_layer)

    @classmethod
    def from_config(
        cls, n_observations: int, n_actions: int, config_path: pathlib.Path
    ):
        config_dict = load_yaml(config_path)
        return cls.from_dict(n_observations, n_actions, config_dict)
