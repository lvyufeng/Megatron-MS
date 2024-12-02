# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib

# from megatron.core.utils import make_viewless_tensor

# try:
#     from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
#     HAVE_PERSIST_LAYER_NORM = True
# except:
#     HAVE_PERSIST_LAYER_NORM = False

# try:
#     from apex.normalization.fused_layer_norm import fused_layer_norm_affine
# except:
#     fused_layer_norm_affine = None


class MixedFusedLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5,
                no_persist_layer_norm=True,
                sequence_parallel=False,
                apply_layernorm_1p=False):
        super(MixedFusedLayerNorm, self).__init__(normalized_shape, eps)

        self.apply_layernorm_1p = apply_layernorm_1p

        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

    def reset_parameters(self):
        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)
