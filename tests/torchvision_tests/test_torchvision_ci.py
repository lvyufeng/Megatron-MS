#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch as ms_torch
import torchvision as ms_torchvision

from ..utils import set_mode_by_env_config, param_compare
set_mode_by_env_config()

def test_deform_conv2d():
    input = torch.rand(4, 3, 10, 10)
    kh, kw = 3, 3
    weight = torch.rand(5, 3, kh, kw)
    offset = torch.rand(4, 2 * kh * kw, 8, 8)
    mask = torch.rand(4, kh * kw, 8, 8)
    out1 = torchvision.ops.deform_conv2d(input, offset, weight, mask=mask)
    out2 = torchvision.ops.deform_conv2d(input, offset, weight)

    ms_input = ms_torch.tensor(input.numpy())
    ms_weight = ms_torch.tensor(weight.numpy())
    ms_offset = ms_torch.tensor(offset.numpy())
    ms_mask = ms_torch.tensor(mask.numpy())
    ms_out1 = ms_torchvision.ops.deform_conv2d(ms_input, ms_offset, ms_weight, mask=ms_mask)
    ms_out2 = ms_torchvision.ops.deform_conv2d(ms_input, ms_offset, ms_weight)

    param_compare(out1, ms_out1)
    param_compare(out2, ms_out2)


if __name__ == '__main__':
    set_mode_by_env_config()
    test_deform_conv2d()
