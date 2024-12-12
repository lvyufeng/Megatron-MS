#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch as ms_torch
import torch
# from ..common_utils import is_test_under_ascend_context
from testing.ut.torchvision.common_utils import is_test_under_ascend_context

def net_test_base(ms_net, torch_net, inputs_shape, eps=1e-5):
    x = np.ones((inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]))
    x = x.astype(np.float32)
    ms_x = ms_torch.tensor(x)
    t_x = torch.tensor(x)

    ms_net.eval()
    torch_net.eval()

    ms_out = ms_net(ms_x)
    t_out = torch_net(t_x)
    print(ms_out[0][1:20], t_out[0][1:20])
    assert np.allclose(ms_out.numpy(), t_out.detach().numpy(), atol=eps)

def net_test_base_no_allclose(ms_net, torch_net, inputs_shape):
    # x = np.random.rand(inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3])
    x = np.zeros((inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    x = x.astype(np.float32)
    ms_x = ms_torch.tensor(x)
    t_x = torch.tensor(x)

    ms_net.eval()
    torch_net.eval()

    ms_out = ms_net(ms_x)
    t_out = torch_net(t_x)
    print(ms_out[0][1:20], t_out[0][1:20])

def net_test_multiple_output(ms_net, torch_net, inputs_shape, eps=1e-5):
    x = np.random.rand(inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3])
    x = x.astype(np.float32)
    ms_x = ms_torch.tensor(x)
    t_x = torch.tensor(x)

    ms_net.eval()
    torch_net.eval()

    ms_out = ms_net(ms_x)
    t_out = torch_net(t_x)
    print(ms_out['out'][0][1:20], t_out['out'][0][1:20])
    assert np.allclose(ms_out['out'][0][1:10].numpy(), t_out['out'][0][1:10].detach().numpy(), rtol=1e-3, atol=eps)

def test_alexnet():
    from torchvision.models import alexnet as alexnet_ms
    from torchvision.models import alexnet as alexnet_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(alexnet_ms(pretrained=True), alexnet_t(pretrained=True), (1, 3, 224, 224), eps=_eps)

def test_resnet18():
    from torchvision.models import resnet18 as resnet18_ms
    from torchvision.models import resnet18 as resnet18_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(resnet18_ms(pretrained=True), resnet18_t(pretrained=True), (1, 3, 224, 224), eps=_eps)

def test_resnet34():
    from torchvision.models import resnet34 as resnet34_ms
    from torchvision.models import resnet34 as resnet34_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-2
    net_test_base(resnet34_ms(pretrained=True), resnet34_t(pretrained=True), (1, 3, 224, 224), eps=_eps)

def test_resnet50():
    from torchvision.models import resnet50 as resnet50_ms
    from torchvision.models import resnet50 as resnet50_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-2
    net_test_base(resnet50_ms(pretrained=True), resnet50_t(pretrained=True), (1, 3, 224, 224), eps=_eps)

def test_resnet101():
    from torchvision.models import resnet101 as resnet101_ms
    from torchvision.models import resnet101 as resnet101_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-2
    net_test_base(resnet101_ms(pretrained=True), resnet101_t(pretrained=True), (1, 3, 224, 224), eps=_eps)

def test_resnet152():
    from torchvision.models import resnet152 as resnet152_ms
    from torchvision.models import resnet152 as resnet152_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-2
    net_test_base(resnet152_ms(pretrained=True), resnet152_t(pretrained=True), (1, 3, 224, 224), eps=_eps)

def test_resnext50_32x4d():
    from torchvision.models import resnext50_32x4d as resnext50_32x4d_ms
    from torchvision.models import resnext50_32x4d as resnext50_32x4d_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-3
    net_test_base(resnext50_32x4d_ms(pretrained=True), resnext50_32x4d_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_resnext101_32x8d():
    from torchvision.models import resnext101_32x8d as resnext101_32x8d_ms
    from torchvision.models import resnext101_32x8d as resnext101_32x8d_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-3
    net_test_base(resnext101_32x8d_ms(pretrained=True), resnext101_32x8d_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_wide_resnet50_2():
    from torchvision.models import wide_resnet50_2 as wide_resnet50_2_ms
    from torchvision.models import wide_resnet50_2 as wide_resnet50_2_t
    net_test_base(wide_resnet50_2_ms(pretrained=True), wide_resnet50_2_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-2)

def test_wide_resnet101_2():
    from torchvision.models import wide_resnet101_2 as wide_resnet101_2_ms
    from torchvision.models import wide_resnet101_2 as wide_resnet101_2_t
    net_test_base(wide_resnet101_2_ms(pretrained=True), wide_resnet101_2_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-2)

def test_vgg11():
    from torchvision.models import vgg11 as vgg11_ms
    from torchvision.models import vgg11 as vgg11_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg11_ms(pretrained=True), vgg11_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg13():
    from torchvision.models import vgg13 as vgg13_ms
    from torchvision.models import vgg13 as vgg13_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg13_ms(pretrained=True), vgg13_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg16():
    from torchvision.models import vgg16 as vgg16_ms
    from torchvision.models import vgg16 as vgg16_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg16_ms(pretrained=True), vgg16_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg19():
    from torchvision.models import vgg19 as vgg19_ms
    from torchvision.models import vgg19 as vgg19_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg19_ms(pretrained=True), vgg19_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg11_bn():
    from torchvision.models import vgg11_bn as vgg11_bn_ms
    from torchvision.models import vgg11_bn as vgg11_bn_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg11_bn_ms(pretrained=True), vgg11_bn_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg13_bn():
    from torchvision.models import vgg13_bn as vgg13_bn_ms
    from torchvision.models import vgg13_bn as vgg13_bn_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg13_bn_ms(pretrained=True), vgg13_bn_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg16_bn():
    from torchvision.models import vgg16_bn as vgg16_bn_ms
    from torchvision.models import vgg16_bn as vgg16_bn_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg16_bn_ms(pretrained=True), vgg16_bn_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_vgg19_bn():
    from torchvision.models import vgg19_bn as vgg19_bn_ms
    from torchvision.models import vgg19_bn as vgg19_bn_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vgg19_bn_ms(pretrained=True), vgg19_bn_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)


def test_squeezenet1_0():
    from torchvision.models import squeezenet1_0 as squeezenet1_0_ms
    from torchvision.models import squeezenet1_0 as squeezenet1_0_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(squeezenet1_0_ms(pretrained=True), squeezenet1_0_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_squeezenet1_1():
    from torchvision.models import squeezenet1_1 as squeezenet1_1_ms
    from torchvision.models import squeezenet1_1 as squeezenet1_1_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(squeezenet1_1_ms(pretrained=True), squeezenet1_1_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_inception_v3():
    from torchvision.models import inception_v3 as inception_v3_ms
    from torchvision.models import inception_v3 as inception_v3_t
    if is_test_under_ascend_context():
        _eps = 6e-2
    else:
        _eps = 1e-3
    net_test_base_no_allclose(inception_v3_ms(pretrained=True), inception_v3_t(pretrained=True),
                  (1, 3, 224, 224))

def test_densenet121():
    from torchvision.models import densenet121 as densenet121_ms
    from torchvision.models import densenet121 as densenet121_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base_no_allclose(densenet121_ms(pretrained=True), densenet121_t(pretrained=True),
                  (1, 3, 224, 224))

def test_densenet161():
    from torchvision.models import densenet161 as densenet161_ms
    from torchvision.models import densenet161 as densenet161_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-3
    net_test_base_no_allclose(densenet161_ms(pretrained=True), densenet161_t(pretrained=True),
                  (1, 3, 224, 224))

def test_densenet169():
    from torchvision.models import densenet169 as densenet169_ms
    from torchvision.models import densenet169 as densenet169_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base_no_allclose(densenet169_ms(pretrained=True), densenet169_t(pretrained=True),
                  (1, 3, 224, 224))

def test_densenet201():
    from torchvision.models import densenet201 as densenet201_ms
    from torchvision.models import densenet201 as densenet201_t
    net_test_base_no_allclose(densenet201_ms(pretrained=True), densenet201_t(pretrained=True),
                  (1, 3, 224, 224))

def test_mobilentv2():
    from torchvision.models import mobilenet_v2 as mobilenet_v2_ms
    from torchvision.models import mobilenet_v2 as mobilenet_v2_t
    if is_test_under_ascend_context():
        _eps = 3e-2
    else:
        _eps = 1e-3
    net_test_base(mobilenet_v2_ms(pretrained=True), mobilenet_v2_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_mobilentv3_large():
    from torchvision.models import mobilenet_v3_large as mobilenet_v3_large_ms
    from torchvision.models import mobilenet_v3_large as mobilenet_v3_large_t
    if is_test_under_ascend_context():
        _eps = 6e-2
    else:
        _eps = 1e-3
    net_test_base(mobilenet_v3_large_ms(pretrained=True), mobilenet_v3_large_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_mobilenet_v3_small():
    from torchvision.models import mobilenet_v3_small as mobilenet_v3_small_ms
    from torchvision.models import mobilenet_v3_small as mobilenet_v3_small_t
    if is_test_under_ascend_context():
        _eps = 8e-2
    else:
        _eps = 1e-3
    net_test_base(mobilenet_v3_small_ms(pretrained=True), mobilenet_v3_small_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_mnasnet0_5():
    from torchvision.models import mnasnet0_5 as mnasnet0_5_ms
    from torchvision.models import mnasnet0_5 as mnasnet0_5_t
    if is_test_under_ascend_context():
        _eps = 5e-2
    else:
        _eps = 1e-3
    net_test_base(mnasnet0_5_ms(pretrained=True), mnasnet0_5_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_mnasnet1_0():
    from torchvision.models import mnasnet1_0 as mnasnet1_0_ms
    from torchvision.models import mnasnet1_0 as mnasnet1_0_t
    if is_test_under_ascend_context():
        _eps = 6e-2
    else:
        _eps = 1e-3
    net_test_base(mnasnet1_0_ms(pretrained=True), mnasnet1_0_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)


def test_shufflenet_v2_x0_5():
    from torchvision.models import shufflenet_v2_x0_5 as shufflenet_v2_x0_5_ms
    from torchvision.models import shufflenet_v2_x0_5 as shufflenet_v2_x0_5_t
    if is_test_under_ascend_context():
        _eps = 8e-2
    else:
        _eps = 1e-3
    net_test_base(shufflenet_v2_x0_5_ms(pretrained=True), shufflenet_v2_x0_5_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_shufflenet_v2_x1_0():
    from torchvision.models import shufflenet_v2_x1_0 as shufflenet_v2_x1_0_ms
    from torchvision.models import shufflenet_v2_x1_0 as shufflenet_v2_x1_0_t
    if is_test_under_ascend_context():
        _eps = 8e-2
    else:
        _eps = 1e-3
    net_test_base(shufflenet_v2_x1_0_ms(pretrained=True), shufflenet_v2_x1_0_t(pretrained=True),
                  (1, 3, 224, 224), eps=_eps)

def test_deeplabv3_mobilenet():
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplabv3_mobilenet_v3_large_ms
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as deeplabv3_mobilenet_v3_large_t
    net_test_multiple_output(deeplabv3_mobilenet_v3_large_ms(pretrained=True), deeplabv3_mobilenet_v3_large_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-1)

def test_deeplabv3_resnet50():
    from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3_resnet50_ms
    from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3_resnet50_t
    net_test_multiple_output(deeplabv3_resnet50_ms(pretrained=True), deeplabv3_resnet50_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-1)

def test_deeplabv3_resnet101():
    from torchvision.models.segmentation import deeplabv3_resnet101 as deeplabv3_resnet101_ms
    from torchvision.models.segmentation import deeplabv3_resnet101 as deeplabv3_resnet101_t
    net_test_multiple_output(deeplabv3_resnet101_ms(pretrained=True), deeplabv3_resnet101_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-0)

def test_fcn_resnet50():
    from torchvision.models.segmentation import fcn_resnet50 as fcn_resnet50_ms
    from torchvision.models.segmentation import fcn_resnet50 as fcn_resnet50_t
    net_test_multiple_output(fcn_resnet50_ms(pretrained=True), fcn_resnet50_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-0)

def test_fcn_resnet101():
    from torchvision.models.segmentation import fcn_resnet101 as fcn_resnet101_ms
    from torchvision.models.segmentation import fcn_resnet101 as fcn_resnet101_t
    net_test_multiple_output(fcn_resnet101_ms(pretrained=True), fcn_resnet101_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-0)

def test_lraspp_mobilenet_v3_large():
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large as lraspp_mobilenet_v3_large_ms
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large as lraspp_mobilenet_v3_large_t
    net_test_multiple_output(lraspp_mobilenet_v3_large_ms(pretrained=True), lraspp_mobilenet_v3_large_t(pretrained=True),
                  (1, 3, 224, 224), eps=1e-0)

def test_fasterrcnn_mobilenet_v3_large_fpn():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as fasterrcnn_mobilenet_v3_large_fpn_ms
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as fasterrcnn_mobilenet_v3_large_fpn_t
    import torch
    import torch as mtorch
    net_t = fasterrcnn_mobilenet_v3_large_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print("Torch out", out_t)
    net_m = fasterrcnn_mobilenet_v3_large_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print("MindTorch out", out_m)

def test_fasterrcnn_mobilenet_v3_large_320_fpn():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn as fasterrcnn_mobilenet_v3_large_320_fpn_ms
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn as fasterrcnn_mobilenet_v3_large_320_fpn_t
    import torch
    import torch as mtorch
    x = np.random.rand(3, 300, 400)
    y = np.random.rand(3, 500, 400)

    net_t = fasterrcnn_mobilenet_v3_large_320_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.tensor(x), torch.tensor(y)])
    print(out_t)
    net_m = fasterrcnn_mobilenet_v3_large_320_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.tensor(x), mtorch.tensor(y)])
    print(out_m)

def test_fasterrcnn_resnet50_fpn():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_ms
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_t
    import torch
    import torch as mtorch
    net_t = fasterrcnn_resnet50_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = fasterrcnn_resnet50_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)

def test_keypointrcnn_resnet50_fpn():
    from torchvision.models.detection import keypointrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_ms
    from torchvision.models.detection import keypointrcnn_resnet50_fpn as keypointrcnn_resnet50_fpn_t
    import torch
    import torch as mtorch
    net_t = keypointrcnn_resnet50_fpn_t(pretrained='legacy')
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = fasterrcnn_resnet50_fpn_ms(pretrained='legacy')
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)


def test_keypointrcnn_resnet50_coco():
    from torchvision.models.detection import keypointrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn_ms
    from torchvision.models.detection import keypointrcnn_resnet50_fpn as keypointrcnn_resnet50_fpn_t
    import torch
    import torch as mtorch
    net_t = keypointrcnn_resnet50_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = fasterrcnn_resnet50_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)

def test_maskrcnn_resnet50_fpn():
    from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn_resnet50_fpn_ms
    from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn_resnet50_fpn_t
    import torch
    import torch as mtorch
    net_t = maskrcnn_resnet50_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = maskrcnn_resnet50_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)

def test_retinanet_resnet50_fpn():
    from torchvision.models.detection import retinanet_resnet50_fpn as retinanet_resnet50_fpn_ms
    from torchvision.models.detection import retinanet_resnet50_fpn as retinanet_resnet50_fpn_t
    import torch
    import torch as mtorch
    net_t = retinanet_resnet50_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = retinanet_resnet50_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)

def test_ssdlite320_mobilenet_v3_large():
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large as ssdlite320_mobilenet_v3_large_ms
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large as ssdlite320_mobilenet_v3_large_t
    import torch
    import torch as mtorch
    net_t = ssdlite320_mobilenet_v3_large_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = ssdlite320_mobilenet_v3_large_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)

def test_ssd300_vgg16():
    from torchvision.models.detection import ssd300_vgg16 as ssd300_vgg16_ms
    from torchvision.models.detection import ssd300_vgg16 as ssd300_vgg16_t
    import torch
    import torch as mtorch
    net_t = ssd300_vgg16_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
    print(out_t)
    net_m = ssd300_vgg16_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.rand(3, 300, 400), mtorch.rand(3, 500, 400)])
    print(out_m)

def test_r3d_18():
    from torchvision.models.video import r3d_18 as r3d_18_ms
    from torchvision.models.video import r3d_18 as r3d_18_t
    import torch
    import torch as mtorch
    net_t = r3d_18_t(pretrained=True)
    net_t.eval()
    out_t = net_t(torch.rand(64, 3, 32, 32, 32))
    print(out_t)
    net_m = r3d_18_ms(pretrained=True)
    net_m.eval()
    out_m = net_m(mtorch.rand(64, 3, 32, 32, 32))
    print(out_m)

def test_r2plus1d_18():
    from torchvision.models.video import r2plus1d_18 as r2plus1d_18_ms
    from torchvision.models.video import r2plus1d_18 as r2plus1d_18_t
    import torch
    import torch as mtorch
    net_t = r2plus1d_18_t(pretrained=True)
    net_t.eval()
    out_t = net_t(torch.rand(64, 3, 32, 32, 32))
    print(out_t)
    net_m = r2plus1d_18_ms(pretrained=True)
    net_m.eval()
    out_m = net_m(mtorch.rand(64, 3, 32, 32, 32))
    print(out_m)

def test_mc3_18():
    from torchvision.models.video import mc3_18 as mc3_18_ms
    from torchvision.models.video import mc3_18 as mc3_18_t
    import torch
    import torch as mtorch
    net_t = mc3_18_t(pretrained=True)
    net_t.eval()
    out_t = net_t(torch.rand(64, 3, 32, 32, 32))
    print(out_t)
    net_m = mc3_18_ms(pretrained=True)
    net_m.eval()
    out_m = net_m(mtorch.rand(64, 3, 32, 32, 32))
    print(out_m)

def test_convnext():
    from torchvision.models.convnext import convnext_base as convnext_base_ms
    from torchvision.models.convnext import convnext_base as convnext_base_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(convnext_base_ms(pretrained=True), convnext_base_t(pretrained=True), (64, 3, 32, 32), eps=_eps)

def test_efficientnet():
    from torchvision.models.efficientnet import efficientnet_b0 as efficientnet_b0_ms
    from torchvision.models.efficientnet import efficientnet_b0 as efficientnet_b0_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(efficientnet_b0_ms(pretrained=True), efficientnet_b0_t(pretrained=True), (64, 3, 32, 32), eps=_eps)

def test_googlenet():
    from torchvision.models.googlenet import googlenet as googlenet_ms
    from torchvision.models.googlenet import googlenet as googlenet_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(googlenet_ms(pretrained=True), googlenet_t(pretrained=True), (64, 3, 32, 32), eps=_eps)

def test_regnet():
    from torchvision.models.regnet import regnet_x_1_6gf as regnet_x_1_6gf_ms
    from torchvision.models.regnet import regnet_x_1_6gf as regnet_x_1_6gf_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(regnet_x_1_6gf_ms(pretrained=True), regnet_x_1_6gf_t(pretrained=True), (64, 3, 32, 32), eps=_eps)

def test_swint():
    from torchvision.models.swin_transformer import swin_t as swin_t_ms
    from torchvision.models.swin_transformer import swin_t as swin_t_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(swin_t_ms(), swin_t_t(), (64, 3, 32, 32), eps=_eps)

def test_vision_transformer():
    from torchvision.models.vision_transformer import vit_b_16 as vit_b_16_ms
    from torchvision.models.vision_transformer import vit_b_16 as vit_b_16_t
    if is_test_under_ascend_context():
        _eps = 1e-2
    else:
        _eps = 1e-3
    net_test_base(vit_b_16_ms(pretrained=True), vit_b_16_t(pretrained=True), (64, 3, 224, 224), eps=_eps)

def test_fcos_resnet50_fpn():
    from torchvision.models.detection import fcos_resnet50_fpn as fcos_resnet50_fpn_ms
    from torchvision.models.detection import fcos_resnet50_fpn as fcos_resnet50_fpn_t
    import torch
    import torch as mtorch
    x = np.random.rand(3, 300, 400)
    y = np.random.rand(3, 500, 400)
    net_t = fcos_resnet50_fpn_t(pretrained=True)
    net_t.eval()
    out_t = net_t([torch.tensor(x, dtype=torch.float32), torch.tensor(y)])
    print("Torch out", out_t)
    net_m = fcos_resnet50_fpn_ms(pretrained=True)
    net_m.eval()
    out_m = net_m([mtorch.tensor(x), mtorch.tensor(y)])
    print("MindTorch out", out_m)

def test_raft_small():
    from torchvision.models.optical_flow import raft_small as raft_small_ms
    from torchvision.models.optical_flow import raft_small as raft_small_t
    x = np.random.rand(1, 3, 224, 224)
    x = x.astype(np.float32)
    ms_x = ms_torch.tensor(x)
    t_x = torch.tensor(x)

    y = np.random.rand(1, 3, 224, 224)
    y = y.astype(np.float32)
    ms_y = ms_torch.tensor(y)
    t_y = torch.tensor(y)

    ms_net = raft_small_ms()
    torch_net = raft_small_t()

    ms_net.eval()
    torch_net.eval()

    # ms_out = ms_net(ms_x, ms_y)
    t_out = torch_net(t_x, t_y)
    print(t_out)
    # assert np.allclose(ms_out.numpy(), t_out.detach().numpy(), atol=eps)


if __name__ == '__main__':
    test_convnext()
    test_efficientnet()
    test_googlenet()
    test_regnet()
    # TODO Swin Transformer model needs to be tested on MindSpore 2.3.
    # test_swint()
    # test_vision_transformer()
    # TODO fcos_resnet50_fpn result is correct, but the output dimensions are not the same as torch.
    test_fcos_resnet50_fpn()
    # TODO Unsupported op InstanceNorm on CPU.
    # test_raft_small()
    test_alexnet()
    test_resnet18()
    test_resnet34()
    test_resnet50()
    test_resnet101()
    test_resnet152()
    test_resnext50_32x4d()
    test_resnext101_32x8d()
    test_wide_resnet50_2()
    test_wide_resnet101_2()
    test_vgg11()
    test_vgg13()
    test_vgg16()
    test_vgg19()
    test_vgg11_bn()
    test_vgg13_bn()
    test_vgg16_bn()
    test_vgg19_bn()
    test_squeezenet1_0()
    test_squeezenet1_1()
    # TODO inception result is incorrect, Because ms.ops.avgpool this function is incorrectly evaluated when C is greater than 8.
    test_inception_v3()
    # TODO densenet result is incorrect, Because ms.ops.adaptive_avg_pool2d this function is incorrectly evaluated
    test_densenet121()
    test_densenet161()
    test_densenet169()
    test_densenet201()
    test_mobilentv2()
    test_mobilentv3_large()
    test_mobilenet_v3_small()
    test_mnasnet0_5()
    test_mnasnet1_0()
    test_shufflenet_v2_x0_5()
    test_shufflenet_v2_x1_0()
    test_deeplabv3_mobilenet()
    test_deeplabv3_resnet50()
    test_deeplabv3_resnet101()
    test_fcn_resnet50()
    test_fcn_resnet101()
    test_lraspp_mobilenet_v3_large()
    # TODO fasterrcnn keypointrcnn result is correct, If the output data is empty, post-processing will not be possible.
    # test_fasterrcnn_mobilenet_v3_large_fpn()
    # test_fasterrcnn_mobilenet_v3_large_320_fpn()
    # test_fasterrcnn_resnet50_fpn()
    # test_keypointrcnn_resnet50_fpn()
    # test_keypointrcnn_resnet50_coco()
    # TODO maskrcnn Because pad cannot pad empty tensors. If the output data is empty, post-processing will not be possible.
    # test_maskrcnn_resnet50_fpn()
    test_retinanet_resnet50_fpn()
    test_ssdlite320_mobilenet_v3_large()
    test_ssd300_vgg16()
    test_r3d_18()
    test_r2plus1d_18()
    test_mc3_18()
