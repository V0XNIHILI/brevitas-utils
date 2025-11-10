import torch

from brevitas_utils.custom_clamps.ignore_overflow_clamp import ignore_overflow_clamp


def test_ignore_overflow_clamp():
    inputs = torch.tensor([1.0, 0.0, 16.0, 15.0, 17.0, 18.0], requires_grad=True)
    inputs2 = torch.tensor([1.0, 0.0, 16.0, 15.0, 17.0, 18.0], requires_grad=True)

    outputs = ignore_overflow_clamp(inputs, torch.tensor(0), torch.tensor(15))
    outputs_clamped = ignore_overflow_clamp(inputs2, torch.tensor(0), torch.tensor(15), gradient_outside_range=0.0)

    assert torch.allclose(outputs, outputs_clamped), f"Clamping does not work correctly"

    assert torch.allclose(outputs, torch.tensor([1.0, 0.0, 0.0, 15., 1.0, 2.0])), f"Overflow clamping does not work"

    torch.sum(outputs).backward()

    assert torch.allclose(inputs.grad, torch.ones_like(inputs)), f"Gradient is not correct"
    
    torch.sum(outputs_clamped).backward()

    assert torch.allclose(inputs2.grad, torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])), f"Gradient is not correct"
