import torch.nn as nn


def replace_node_module(top_level_module: nn.Module, name: str,
                        new_module: nn.Module):
    # Example name: "tcn.0.0.temp_layer1.4"

    # Split string into list of strings
    name_list = name.split(".")

    for name in name_list[:-1]:
        top_level_module = getattr(top_level_module, name)

    setattr(top_level_module, name_list[-1], new_module)
