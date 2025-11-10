import torch

from brevitas_utils.custom_quantizers.pot4_weight_per_tensor_fixed_point import potquant


def test_potquant_values():
    for bit_width in range(4, 5):
        inputs = 2**torch.tensor(range(0, 2**(bit_width-1))).float()

        inputs = torch.cat([inputs, -inputs])

        print(inputs)

        outputs = potquant(inputs, bit_width)

        print(outputs)

        print(len(torch.unique(outputs)))

        assert torch.allclose(outputs, inputs), f"Power-of-two valued number do not match for bit width {bit_width}"

        if bit_width < 9:
            assert len(torch.unique(outputs)) == 2**bit_width, f"Number of possible values do not match for bit width {bit_width}"


def test_potquant_clamp():
    inputs = torch.tensor([256.0, -256.0])

    outputs = potquant(inputs, 4)

    assert torch.max(torch.abs(outputs)) <= 128.0, f"Clamping does not work"
