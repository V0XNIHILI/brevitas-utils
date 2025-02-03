"""Copied from: https://github.com/Xilinx/brevitas/blob/4617f7bd136e96fa21c7f76e3c7e2e37fe563837/src/brevitas_examples/llm/llm_quant/prepare_for_quantize.py#L28
to be able to import this function without requiring the transformers library."""

import torch


@torch.no_grad()
def add_zero_bias_to_linear(model: torch.nn.Module) -> torch.nn.Module:
    for _, module in model.named_modules():
        if type(module) == torch.nn.Linear:
            if module.bias is None:
                module.register_parameter(
                    "bias",
                    torch.nn.Parameter(
                        torch.zeros((module.weight.shape[0],),
                                    device=module.weight.device,
                                    dtype=module.weight.dtype)),
                )
    return model
