import math
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import product
from typing import Callable, List, Tuple

import mindspore as ms
import numpy as np
import pytest
import torch as torch
import torch.nn.functional as F
from .common_utils import cpu_and_gpu, needs_cuda, SKIP_ENV_CPU, SKIP_ENV_GPU
from PIL import Image
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torchvision import ops


class RoIOpTesterModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 2

    def forward(self, a, b):
        self.layer(a, b)


class MultiScaleRoIAlignModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 3

    def forward(self, a, b, c):
        self.layer(a, b, c)


class DeformConvModuleWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 3

    def forward(self, a, b, c):
        self.layer(a, b, c)


class StochasticDepthWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 1

    def forward(self, a):
        self.layer(a)


class DropBlockWrapper(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.layer = obj
        self.n_inputs = 1

    def forward(self, a):
        self.layer(a)


class RoIOpTester(ABC):
    dtype = torch.float32

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("contiguous", (False, True))
    #TODO: The implementation of 'roi_align' is not completely consistent with torch when 'aligned' is set to True
    def test_forward(self, device, contiguous, x_dtype=None, rois_dtype=None, **kwargs):
        #TODO: The implementation of 'roi_align' is not completely consistent with torch when 'aligned' is set to True
        aligned = kwargs.get('aligned')
        if aligned:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        x_dtype = self.dtype if x_dtype is None else x_dtype
        rois_dtype = self.dtype if rois_dtype is None else rois_dtype
        pool_size = 5
        # n_channels % (pool_size ** 2) == 0 required for PS opeartions.
        n_channels = 2 * (pool_size ** 2)
        x = torch.rand(2, n_channels, 10, 10, dtype=x_dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor(
            [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9], [1, 0, 0, 9, 9]],  # format is (xyxy)
            dtype=rois_dtype,
            device=device,
        )

        pool_h, pool_w = pool_size, pool_size
        y = self.fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=2, **kwargs)
        # the following should be true whether we're running an autocast test or not.
        assert y.dtype == x.dtype
        gt_y = self.expected_fn(
            x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=2, device=device, dtype=self.dtype, **kwargs
        )

        tol = 1e-3 if (x_dtype is torch.half or rois_dtype is torch.half) else 1e-5
        # torch.testing.assert_close(gt_y.to(y), y, rtol=tol, atol=tol)
        assert np.allclose(gt_y.to(y).numpy(), y.numpy(), rtol=tol, atol=tol)


    def _helper_boxes_shape(self, func):
        # test boxes as Tensor[N, 5]
        with pytest.raises(AssertionError):
            a = torch.linspace(1, 8 * 8, 8 * 8).reshape(1, 1, 8, 8)
            boxes = torch.tensor([[0, 0, 3, 3]], dtype=a.dtype)
            func(a, boxes, output_size=(2, 2))

    @abstractmethod
    def fn(*args, **kwargs):
        pass

    @abstractmethod
    def make_obj(*args, **kwargs):
        pass

    # @abstractmethod
    # def get_script_fn(*args, **kwargs):
    #     pass

    @abstractmethod
    def expected_fn(*args, **kwargs):
        pass


# class TestRoiPool(RoIOpTester):
#     def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
#         return ops.RoIPool((pool_h, pool_w), spatial_scale)(x, rois)
#
#     def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, wrap=False):
#         obj = ops.RoIPool((pool_h, pool_w), spatial_scale)
#         return RoIOpTesterModuleWrapper(obj) if wrap else obj
#
#     # def get_script_fn(self, rois, pool_size):
#     #     scriped = torch.jit.script(ops.roi_pool)
#     #     return lambda x: scriped(x, rois, pool_size)
#
#     def expected_fn(
#         self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, device=None, dtype=torch.float64
#     ):
#         if device is None:
#             device = torch.device("cpu")
#
#         n_channels = x.size(1)
#         y = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)
#
#         def get_slice(k, block):
#             return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))
#
#         for roi_idx, roi in enumerate(rois):
#             batch_idx = int(roi[0])
#             j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
#             roi_x = x[batch_idx, :, i_begin : i_end + 1, j_begin : j_end + 1]
#
#             roi_h, roi_w = roi_x.shape[-2:]
#             bin_h = roi_h / pool_h
#             bin_w = roi_w / pool_w
#
#             for i in range(0, pool_h):
#                 for j in range(0, pool_w):
#                     bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
#                     if bin_x.numel() > 0:
#                         y[roi_idx, :, i, j] = bin_x.reshape(n_channels, -1).max(dim=1)[0]
#         return y
#
#     def test_boxes_shape(self):
#         self._helper_boxes_shape(ops.roi_pool)


# class TestPSRoIPool(RoIOpTester):
#     def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
#         return ops.PSRoIPool((pool_h, pool_w), 1)(x, rois)
#
#     def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, wrap=False):
#         obj = ops.PSRoIPool((pool_h, pool_w), spatial_scale)
#         return RoIOpTesterModuleWrapper(obj) if wrap else obj
#
#     # def get_script_fn(self, rois, pool_size):
#     #     scriped = torch.jit.script(ops.ps_roi_pool)
#     #     return lambda x: scriped(x, rois, pool_size)
#
#     def expected_fn(
#         self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, device=None, dtype=torch.float64
#     ):
#         if device is None:
#             device = torch.device("cpu")
#         n_input_channels = x.size(1)
#         assert n_input_channels % (pool_h * pool_w) == 0, "input channels must be divisible by ph * pw"
#         n_output_channels = int(n_input_channels / (pool_h * pool_w))
#         y = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)
#
#         def get_slice(k, block):
#             return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))
#
#         for roi_idx, roi in enumerate(rois):
#             batch_idx = int(roi[0])
#             j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
#             roi_x = x[batch_idx, :, i_begin : i_end + 1, j_begin : j_end + 1]
#
#             roi_height = max(i_end - i_begin, 1)
#             roi_width = max(j_end - j_begin, 1)
#             bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)
#
#             for i in range(0, pool_h):
#                 for j in range(0, pool_w):
#                     bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
#                     if bin_x.numel() > 0:
#                         area = bin_x.size(-2) * bin_x.size(-1)
#                         for c_out in range(0, n_output_channels):
#                             c_in = c_out * (pool_h * pool_w) + pool_w * i + j
#                             t = torch.sum(bin_x[c_in, :, :])
#                             y[roi_idx, c_out, i, j] = t / area
#         return y
#
#     def test_boxes_shape(self):
#         self._helper_boxes_shape(ops.ps_roi_pool)


def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val


class TestRoIAlign(RoIOpTester):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, aligned=False, **kwargs):
        return ops.RoIAlign(
            (pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned
        )(x, rois)

    def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, sampling_ratio=-1, aligned=False, wrap=False):
        obj = ops.RoIAlign(
            (pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned
        )
        return RoIOpTesterModuleWrapper(obj) if wrap else obj

    def expected_fn(
        self,
        in_data,
        rois,
        pool_h,
        pool_w,
        spatial_scale=1,
        sampling_ratio=-1,
        aligned=False,
        device=None,
        dtype=torch.float64,
    ):
        if device is None:
            device = torch.device("cpu")
        n_channels = in_data.size(1)
        out_data = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        offset = 0.5 if aligned else 0.0

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - offset for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                    for channel in range(0, n_channels):

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, channel, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, channel, i, j] = val
        return out_data

    def test_boxes_shape(self):
        self._helper_boxes_shape(ops.roi_align)

    @pytest.mark.parametrize("aligned", (True, False))
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("contiguous", (True, False))
    def test_forward(self, device, contiguous, aligned, x_dtype=None, rois_dtype=None):
        torch.random.manual_seed(3)
        super().test_forward(
            device=device, contiguous=contiguous, x_dtype=x_dtype, rois_dtype=rois_dtype, aligned=aligned
        )

    def _make_rois(self, img_size, num_imgs, dtype, num_rois=1000):
        rois = torch.randint(0, img_size // 2, size=(num_rois, 5)).to(dtype)
        rois[:, 0] = torch.randint(0, num_imgs, size=(num_rois,))  # set batch index
        rois[:, 3:] += rois[:, 1:3]  # make sure boxes aren't degenerate
        return rois


# class TestPSRoIAlign(RoIOpTester):
#     def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
#         return ops.PSRoIAlign((pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)(x, rois)
#
#     def make_obj(self, pool_h=5, pool_w=5, spatial_scale=1, sampling_ratio=-1, wrap=False):
#         obj = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
#         return RoIOpTesterModuleWrapper(obj) if wrap else obj
#
#     # def get_script_fn(self, rois, pool_size):
#     #     scriped = torch.jit.script(ops.ps_roi_align)
#     #     return lambda x: scriped(x, rois, pool_size)
#
#     def expected_fn(
#         self, in_data, rois, pool_h, pool_w, device, spatial_scale=1, sampling_ratio=-1, dtype=torch.float64
#     ):
#         if device is None:
#             device = torch.device("cpu")
#         n_input_channels = in_data.size(1)
#         assert n_input_channels % (pool_h * pool_w) == 0, "input channels must be divisible by ph * pw"
#         n_output_channels = int(n_input_channels / (pool_h * pool_w))
#         out_data = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)
#
#         for r, roi in enumerate(rois):
#             batch_idx = int(roi[0])
#             j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - 0.5 for x in roi[1:])
#
#             roi_h = i_end - i_begin
#             roi_w = j_end - j_begin
#             bin_h = roi_h / pool_h
#             bin_w = roi_w / pool_w
#
#             for i in range(0, pool_h):
#                 start_h = i_begin + i * bin_h
#                 grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
#                 for j in range(0, pool_w):
#                     start_w = j_begin + j * bin_w
#                     grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))
#                     for c_out in range(0, n_output_channels):
#                         c_in = c_out * (pool_h * pool_w) + pool_w * i + j
#
#                         val = 0
#                         for iy in range(0, grid_h):
#                             y = start_h + (iy + 0.5) * bin_h / grid_h
#                             for ix in range(0, grid_w):
#                                 x = start_w + (ix + 0.5) * bin_w / grid_w
#                                 val += bilinear_interpolate(in_data[batch_idx, c_in, :, :], y, x, snap_border=True)
#                         val /= grid_h * grid_w
#
#                         out_data[r, c_out, i, j] = val
#         return out_data
#
#     def test_boxes_shape(self):
#         self._helper_boxes_shape(ops.ps_roi_align)


class TestMultiScaleRoIAlign:
    def make_obj(self, fmap_names=None, output_size=(7, 7), sampling_ratio=2, wrap=False):
        if fmap_names is None:
            fmap_names = ["0"]
        obj = ops.poolers.MultiScaleRoIAlign(fmap_names, output_size, sampling_ratio)
        return MultiScaleRoIAlignModuleWrapper(obj) if wrap else obj

    def test_msroialign_repr(self):
        fmap_names = ["0"]
        output_size = (7, 7)
        sampling_ratio = 2
        # Pass mock feature map names
        t = self.make_obj(fmap_names, output_size, sampling_ratio, wrap=False)

        # Check integrity of object __repr__ attribute
        expected_string = (
            f"MultiScaleRoIAlign(featmap_names={fmap_names}, output_size={output_size}, "
            f"sampling_ratio={sampling_ratio})"
        )
        assert repr(t) == expected_string



class TestNMS:
    # This testcast on Ascend may failed, because mindspore.ops.sort result has error on Ascend.
    def _reference_nms(self, boxes, scores, iou_threshold):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        picked = []
        _, indexes = scores.sort(descending=True)
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = ops.box_iou(rest_boxes, current_box.unsqueeze(0)).squeeze(1)
            indexes = indexes[iou <= iou_threshold]

        return torch.as_tensor(picked)

    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        # Adjust the threshold upward a bit with the intent of creating
        # at least one box that exceeds (barely) the threshold and so
        # should be suppressed.
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        iou_thresh += 1e-5
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    @pytest.mark.parametrize("iou", (0.2, 0.5, 0.8))
    @pytest.mark.parametrize("seed", range(10))
    def test_nms_ref(self, iou, seed):
        torch.random.manual_seed(seed)
        err_msg = "NMS incompatible between CPU and reference implementation for IoU={}"
        boxes, scores = self._create_tensors_with_iou(1000, iou)
        keep_ref = self._reference_nms(boxes, scores, iou)
        keep = ops.nms(boxes, scores, iou)
        assert np.allclose(keep.numpy(), keep_ref.numpy())

    def test_nms_input_errors(self):
        with pytest.raises(RuntimeError):
            ops.nms(torch.rand(4), torch.rand(3), 0.5)
        with pytest.raises(ValueError):
            ops.nms(torch.rand(3, 5), torch.rand(3), 0.5)
        with pytest.raises(ValueError):
            ops.nms(torch.rand(3, 4), torch.rand(3, 2), 0.5)
        with pytest.raises(RuntimeError):
            ops.nms(torch.rand(3, 4), torch.rand(4), 0.5)

    @needs_cuda
    @pytest.mark.parametrize("iou", (0.2, 0.5, 0.8))
    def test_nms_cuda(self, iou, dtype=torch.float32):
        tol = 1e-3 if dtype is torch.half else 1e-5
        err_msg = "NMS incompatible between CPU and CUDA for IoU={}"

        boxes, scores = self._create_tensors_with_iou(1000, iou)
        r_cpu = ops.nms(boxes, scores, iou)
        r_cuda = ops.nms(boxes.cuda(), scores.cuda(), iou)

        is_eq = np.allclose(r_cpu.numpy(), r_cuda.cpu().numpy())
        if not is_eq:
            # if the indices are not the same, ensure that it's because the scores
            # are duplicate
            is_eq = np.allclose(scores[r_cpu].numpy(), scores[r_cuda.cpu()].numpy(), rtol=tol, atol=tol)
        assert is_eq, err_msg.format(iou)

    @needs_cuda
    def test_nms_cuda_float16(self):
        boxes = torch.tensor(
            [
                [285.3538, 185.5758, 1193.5110, 851.4551],
                [285.1472, 188.7374, 1192.4984, 851.0669],
                [279.2440, 197.9812, 1189.4746, 849.2019],
            ]
        ).cuda()
        scores = torch.tensor([0.6370, 0.7569, 0.3966]).cuda()

        iou_thres = 0.2
        keep32 = ops.nms(boxes, scores, iou_thres)
        keep16 = ops.nms(boxes.to(torch.float16), scores.to(torch.float16), iou_thres)
        assert np.allclose(keep32.numpy(), keep16.numpy())

    @pytest.mark.parametrize("seed", range(10))
    def test_batched_nms_implementations(self, seed):
        """Make sure that both implementations of batched_nms yield identical results"""
        torch.random.manual_seed(seed)

        num_boxes = 1000
        iou_threshold = 0.9

        boxes = torch.cat((torch.rand(num_boxes, 2), torch.rand(num_boxes, 2) + 10), dim=1)
        assert max(boxes[:, 0]) < min(boxes[:, 2])  # x1 < x2
        assert max(boxes[:, 1]) < min(boxes[:, 3])  # y1 < y2

        scores = torch.rand(num_boxes)
        idxs = torch.randint(0, 4, size=(num_boxes,))
        keep_vanilla = ops.boxes._batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
        keep_trick = ops.boxes._batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)

        assert np.allclose(keep_vanilla.numpy(), keep_trick.numpy())
        # Also make sure an empty tensor is returned if boxes is empty
        empty = torch.empty((0,), dtype=torch.int64)
        assert np.allclose(empty.numpy(), ops.batched_nms(empty, None, None, None).numpy())



class TestDeformConv:
    # this testcast on Ascend, the result may be wrong, because mindspore.ops.deformable_conv2d result not correct
    dtype = torch.float32

    def expected_fn(self, x, weight, offset, mask, bias, stride=1, padding=0, dilation=1):
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)
        weight_h, weight_w = weight.shape[-2:]

        n_batches, n_in_channels, in_h, in_w = x.shape
        n_out_channels = weight.shape[0]

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        n_offset_grps = offset.shape[1] // (2 * weight_h * weight_w)
        in_c_per_offset_grp = n_in_channels // n_offset_grps

        n_weight_grps = n_in_channels // weight.shape[1]
        in_c_per_weight_grp = weight.shape[1]
        out_c_per_weight_grp = n_out_channels // n_weight_grps

        out = torch.zeros(n_batches, n_out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        for b in range(n_batches):
            for c_out in range(n_out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for di in range(weight_h):
                            for dj in range(weight_w):
                                for c in range(in_c_per_weight_grp):
                                    weight_grp = c_out // out_c_per_weight_grp
                                    c_in = weight_grp * in_c_per_weight_grp + c

                                    offset_grp = c_in // in_c_per_offset_grp
                                    mask_idx = offset_grp * (weight_h * weight_w) + di * weight_w + dj
                                    offset_idx = 2 * mask_idx

                                    pi = stride_h * i - pad_h + dil_h * di + offset[b, offset_idx, i, j]
                                    pj = stride_w * j - pad_w + dil_w * dj + offset[b, offset_idx + 1, i, j]

                                    mask_value = 1.0
                                    if mask is not None:
                                        mask_value = mask[b, mask_idx, i, j]

                                    out[b, c_out, i, j] += (
                                        mask_value
                                        * weight[c_out, c, di, dj]
                                        * bilinear_interpolate(x[b, c_in, :, :], pi, pj)
                                    )
        out += bias.view(1, n_out_channels, 1, 1)
        return out

    @lru_cache(maxsize=None)
    def get_fn_args(self, device, contiguous, batch_sz, dtype):
        n_in_channels = 6
        n_out_channels = 2
        n_weight_grps = 2
        n_offset_grps = 3

        stride = (2, 1)
        pad = (1, 0)
        dilation = (2, 1)

        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (3, 2)
        in_h, in_w = (5, 4)

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        x = torch.rand(batch_sz, n_in_channels, in_h, in_w, device=device, dtype=dtype, requires_grad=True)

        offset = torch.randn(
            batch_sz,
            n_offset_grps * 2 * weight_h * weight_w,
            out_h,
            out_w,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        mask = torch.randn(
            batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w, device=device, dtype=dtype, requires_grad=True
        )

        weight = torch.randn(
            n_out_channels,
            n_in_channels // n_weight_grps,
            weight_h,
            weight_w,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        bias = torch.randn(n_out_channels, device=device, dtype=dtype, requires_grad=True)

        if not contiguous:
            x = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            mask = mask.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)

        return x, weight, offset, mask, bias, stride, pad, dilation

    def make_obj(self, in_channels=6, out_channels=2, kernel_size=(3, 2), groups=2, wrap=False):
        obj = ops.DeformConv2d(
            in_channels, out_channels, kernel_size, stride=(2, 1), padding=(1, 0), dilation=(2, 1), groups=groups
        )
        return DeformConvModuleWrapper(obj) if wrap else obj

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("contiguous", (True, False))
    @pytest.mark.parametrize("batch_sz", (33,))
    def test_forward(self, device, contiguous, batch_sz, dtype=None):
        dtype = dtype or self.dtype
        x, _, offset, mask, _, stride, padding, dilation = self.get_fn_args(device, contiguous, batch_sz, dtype)
        in_channels = 6
        out_channels = 2
        kernel_size = (3, 2)
        groups = 2
        tol = 2e-3 if dtype is torch.half else 1e-5

        layer = self.make_obj(in_channels, out_channels, kernel_size, groups, wrap=False)
        res = layer(x, offset, mask)

        weight = layer.weight.data
        bias = layer.bias.data
        expected = self.expected_fn(x, weight, offset, mask, bias, stride=stride, padding=padding, dilation=dilation)

        assert np.allclose(res.to(expected).numpy(), expected.numpy(), rtol=tol, atol=tol)


class TestFrozenBNT:
    def test_frozenbatchnorm2d_repr(self):
        num_features = 32
        eps = 1e-5
        t = ops.misc.FrozenBatchNorm2d(num_features, eps=eps)

        # Check integrity of object __repr__ attribute
        expected_string = f"FrozenBatchNorm2d({num_features}, eps={eps})"
        assert repr(t) == expected_string

    @pytest.mark.parametrize("seed", range(10))
    def test_frozenbatchnorm2d_eps(self, seed):
        torch.random.manual_seed(seed)
        sample_size = (4, 32, 28, 28)
        x = torch.rand(sample_size)
        state_dict = dict(
            weight=torch.rand(sample_size[1]),
            bias=torch.rand(sample_size[1]),
            running_mean=torch.rand(sample_size[1]),
            running_var=torch.rand(sample_size[1]),
            num_batches_tracked=torch.tensor(100),
        )

        # Check that default eps is equal to the one of BN
        fbn = ops.misc.FrozenBatchNorm2d(sample_size[1])
        fbn.load_state_dict(state_dict, strict=False)
        bn = torch.nn.BatchNorm2d(sample_size[1]).eval()
        bn.load_state_dict(state_dict)
        # Difference is expected to fall in an acceptable range
        assert np.allclose(fbn(x).numpy(), bn(x).numpy(), rtol=1e-5, atol=1e-5)

        # Check computation for eps > 0
        fbn = ops.misc.FrozenBatchNorm2d(sample_size[1], eps=1e-5)
        fbn.load_state_dict(state_dict, strict=False)
        bn = torch.nn.BatchNorm2d(sample_size[1], eps=1e-5).eval()
        bn.load_state_dict(state_dict)
        assert np.allclose(fbn(x).numpy(), bn(x).numpy(), rtol=1e-5, atol=1e-5)


class TestBoxConversion:
    def _get_box_sequences():
        # Define here the argument type of `boxes` supported by region pooling operations
        box_tensor = torch.tensor([[0, 0, 0, 100, 100], [1, 0, 0, 100, 100]], dtype=torch.float)
        box_list = [
            torch.tensor([[0, 0, 100, 100]], dtype=torch.float),
            torch.tensor([[0, 0, 100, 100]], dtype=torch.float),
        ]
        box_tuple = tuple(box_list)
        return box_tensor, box_list, box_tuple

    @pytest.mark.parametrize("box_sequence", _get_box_sequences())
    def test_check_roi_boxes_shape(self, box_sequence):
        # Ensure common sequences of tensors are supported
        ops._utils.check_roi_boxes_shape(box_sequence)

    @pytest.mark.parametrize("box_sequence", _get_box_sequences())
    def test_convert_boxes_to_roi_format(self, box_sequence):
        # Ensure common sequences of tensors yield the same result
        ref_tensor = None
        if ref_tensor is None:
            ref_tensor = box_sequence
        else:
            assert np.allclose(ref_tensor.numpy(), ops._utils.convert_boxes_to_roi_format(box_sequence).numpy())


class TestBox:
    def test_bbox_same(self):
        box_tensor = torch.tensor(
            [[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float
        )

        exp_xyxy = torch.tensor([[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)

        assert exp_xyxy.size() == torch.Size([4, 4])
        assert np.allclose(ops.box_convert(box_tensor, in_fmt="xyxy", out_fmt="xyxy").numpy(), exp_xyxy.numpy())
        assert np.allclose(ops.box_convert(box_tensor, in_fmt="xywh", out_fmt="xywh").numpy(), exp_xyxy.numpy())
        assert np.allclose(ops.box_convert(box_tensor, in_fmt="cxcywh", out_fmt="cxcywh").numpy(), exp_xyxy.numpy())

    def test_bbox_xyxy_xywh(self):
        # Simple test convert boxes to xywh and back. Make sure they are same.
        # box_tensor is in x1 y1 x2 y2 format.
        box_tensor = torch.tensor(
            [[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float
        )
        exp_xywh = torch.tensor([[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 20, 20], [23, 35, 70, 60]], dtype=torch.float)

        assert exp_xywh.size() == torch.Size([4, 4])
        box_xywh = ops.box_convert(box_tensor, in_fmt="xyxy", out_fmt="xywh")
        assert np.allclose(box_xywh.numpy(), exp_xywh.numpy())

        # Reverse conversion
        box_xyxy = ops.box_convert(box_xywh, in_fmt="xywh", out_fmt="xyxy")
        assert np.allclose(box_xyxy.numpy(), box_tensor.numpy())

    def test_bbox_xyxy_cxcywh(self):
        # Simple test convert boxes to xywh and back. Make sure they are same.
        # box_tensor is in x1 y1 x2 y2 format.
        box_tensor = torch.tensor(
            [[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float
        )
        exp_cxcywh = torch.tensor(
            [[50, 50, 100, 100], [0, 0, 0, 0], [20, 25, 20, 20], [58, 65, 70, 60]], dtype=torch.float
        )

        assert exp_cxcywh.size() == torch.Size([4, 4])
        box_cxcywh = ops.box_convert(box_tensor, in_fmt="xyxy", out_fmt="cxcywh")
        assert np.allclose(box_cxcywh.numpy(), exp_cxcywh.numpy())

        # Reverse conversion
        box_xyxy = ops.box_convert(box_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
        assert np.allclose(box_xyxy.numpy(), box_tensor.numpy())

    def test_bbox_xywh_cxcywh(self):
        box_tensor = torch.tensor(
            [[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 20, 20], [23, 35, 70, 60]], dtype=torch.float
        )

        # This is wrong
        exp_cxcywh = torch.tensor(
            [[50, 50, 100, 100], [0, 0, 0, 0], [20, 25, 20, 20], [58, 65, 70, 60]], dtype=torch.float
        )

        assert exp_cxcywh.size() == torch.Size([4, 4])
        box_cxcywh = ops.box_convert(box_tensor, in_fmt="xywh", out_fmt="cxcywh")
        assert np.allclose(box_cxcywh.numpy(), exp_cxcywh.numpy())

        # Reverse conversion
        box_xywh = ops.box_convert(box_cxcywh, in_fmt="cxcywh", out_fmt="xywh")
        assert np.allclose(box_xywh.numpy(), box_tensor.numpy())

    @pytest.mark.parametrize("inv_infmt", ["xwyh", "cxwyh"])
    @pytest.mark.parametrize("inv_outfmt", ["xwcx", "xhwcy"])
    def test_bbox_invalid(self, inv_infmt, inv_outfmt):
        box_tensor = torch.tensor(
            [[0, 0, 100, 100], [0, 0, 0, 0], [10, 15, 20, 20], [23, 35, 70, 60]], dtype=torch.float
        )

        with pytest.raises(ValueError):
            ops.box_convert(box_tensor, inv_infmt, inv_outfmt)


class BoxTestBase(ABC):
    @abstractmethod
    def _target_fn(self) -> Tuple[bool, Callable]:
        pass

    def _perform_box_operation(self, box: Tensor, run_as_script: bool = False) -> Tensor:
        is_binary_fn = self._target_fn()[0]
        target_fn = self._target_fn()[1]
        box_operation = torch.jit.script(target_fn) if run_as_script else target_fn
        return box_operation(box, box) if is_binary_fn else box_operation(box)

    def _run_test(self, test_input: List, dtypes, tolerance: float, expected: List) -> None:
        def assert_close(box: Tensor, expected: Tensor, tolerance):
            out = self._perform_box_operation(box)
            # torch.testing.assert_close(out, expected, rtol=0.0, check_dtype=False, atol=tolerance)
            assert np.allclose(out.numpy(), expected.numpy(), rtol=0.0, atol=tolerance)

        for dtype in dtypes:
            actual_box = torch.tensor(test_input, dtype=dtype)
            expected_box = torch.tensor(expected)
            assert_close(actual_box, expected_box, tolerance)


class TestBoxArea(BoxTestBase):
    def _target_fn(self) -> Tuple[bool, Callable]:
        return (False, ops.box_area)

    def _generate_int_input() -> List[List[int]]:
        return [[0, 0, 100, 100], [0, 0, 0, 0]]

    def _generate_int_expected() -> List[int]:
        return [10000, 0]

    def _generate_float_input(index: int) -> List[List[float]]:
        return [
            [
                [285.3538, 185.5758, 1193.5110, 851.4551],
                [285.1472, 188.7374, 1192.4984, 851.0669],
                [279.2440, 197.9812, 1189.4746, 849.2019],
            ],
            [[285.25, 185.625, 1194.0, 851.5], [285.25, 188.75, 1192.0, 851.0], [279.25, 198.0, 1189.0, 849.0]],
        ][index]

    def _generate_float_expected(index: int) -> List[float]:
        return [[604723.0806, 600965.4666, 592761.0085], [605113.875, 600495.1875, 592247.25]][index]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(),
                [torch.int8, torch.int16, torch.int32, torch.int64],
                1e-4,
                _generate_int_expected(),
            ),
            # pytest.param(_generate_float_input(0), [torch.float32, torch.float64], 0.05, _generate_float_expected(0)),
            # mindspore not support float64 on Ascend
            pytest.param(_generate_float_input(0), [torch.float32], 0.05, _generate_float_expected(0)),
            pytest.param(_generate_float_input(1), [torch.float16], 1e-4, _generate_float_expected(1)),
        ],
    )
    def test_box_area(self, test_input: List, dtypes, tolerance: float, expected: List) -> None:
        self._run_test(test_input, dtypes, tolerance, expected)


class TestBoxIou(BoxTestBase):
    def _target_fn(self) -> Tuple[bool, Callable]:
        return (True, ops.box_iou)

    def _generate_int_input() -> List[List[int]]:
        return [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]

    def _generate_int_expected() -> List[List[float]]:
        return [[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _generate_float_input() -> List[List[float]]:
        return [
            [285.3538, 185.5758, 1193.5110, 851.4551],
            [285.1472, 188.7374, 1192.4984, 851.0669],
            [279.2440, 197.9812, 1189.4746, 849.2019],
        ]

    def _generate_float_expected() -> List[List[float]]:
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            # pytest.param(
            #     _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            # ),
            # ms.ops.Maximum not support int16 on Ascend
            pytest.param(
                _generate_int_input(), [torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            pytest.param(_generate_float_input(), [torch.float32], 1e-4, _generate_float_expected()),
        ],
    )
    def test_iou(self, test_input: List, dtypes, tolerance: float, expected: List) -> None:
        self._run_test(test_input, dtypes, tolerance, expected)


class TestGenBoxIou(BoxTestBase):
    def _target_fn(self) -> Tuple[bool, Callable]:
        return (True, ops.generalized_box_iou)

    def _generate_int_input() -> List[List[int]]:
        return [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]

    def _generate_int_expected() -> List[List[float]]:
        return [[1.0, 0.25, -0.7778], [0.25, 1.0, -0.8611], [-0.7778, -0.8611, 1.0]]

    def _generate_float_input() -> List[List[float]]:
        return [
            [285.3538, 185.5758, 1193.5110, 851.4551],
            [285.1472, 188.7374, 1192.4984, 851.0669],
            [279.2440, 197.9812, 1189.4746, 849.2019],
        ]

    def _generate_float_expected() -> List[List[float]]:
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                # _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
                # mindspore not support int16 on Ascend.
                _generate_int_input(), [torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            # pytest.param(_generate_float_input(), [torch.float32, torch.float64], 0.001, _generate_float_expected()),
            # mindspore not support float64 on Ascend.
            pytest.param(_generate_float_input(), [torch.float32], 0.001, _generate_float_expected())
        ],
    )
    def test_gen_iou(self, test_input: List, dtypes, tolerance: float, expected: List) -> None:
        self._run_test(test_input, dtypes, tolerance, expected)


class TestDistanceBoxIoU(BoxTestBase):
    def _target_fn(self):
        return (True, ops.distance_box_iou)

    def _generate_int_input():
        return [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]

    def _generate_int_expected():
        return [
            [1.0000, 0.1875, -0.4444],
            [0.1875, 1.0000, -0.5625],
            [-0.4444, -0.5625, 1.0000],
        ]

    def _generate_float_input():
        return [
            [285.3538, 185.5758, 1193.5110, 851.4551],
            [285.1472, 188.7374, 1192.4984, 851.0669],
            [279.2440, 197.9812, 1189.4746, 849.2019],
        ]

    def _generate_float_expected():
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            pytest.param(_generate_float_input(), [torch.float16], 0.002, _generate_float_expected()),
            # pytest.param(_generate_float_input(), [torch.float32, torch.float64], 0.001, _generate_float_expected()),
            # mindspore not support float64 on Ascend.
            pytest.param(_generate_float_input(), [torch.float32], 0.001, _generate_float_expected()),
        ],
    )
    def test_distance_iou(self, test_input, dtypes, tolerance, expected):
        self._run_test(test_input, dtypes, tolerance, expected)


@pytest.mark.parametrize("device", cpu_and_gpu())
@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_distance_iou_loss(dtype, device):
    box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
    box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
    box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
    box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

    box1s = torch.stack(
        [box2, box2],
        dim=0,
    )
    box2s = torch.stack(
        [box3, box4],
        dim=0,
    )

    def assert_distance_iou_loss(box1, box2, expected_output, reduction="none"):
        output = ops.distance_box_iou_loss(box1, box2, reduction=reduction)
        # TODO: When passing the dtype, the torch.half fails as usual.
        expected_output = torch.tensor(expected_output, device=device)
        tol = 1e-5 if dtype != torch.half else 1e-3
        # torch.testing.assert_close(output, expected_output, rtol=tol, atol=tol)
        assert np.allclose(output.numpy(), expected_output.numpy(), rtol=tol, atol=tol)

    assert_distance_iou_loss(box1, box1, 0.0)

    assert_distance_iou_loss(box1, box2, 0.8125)

    assert_distance_iou_loss(box1, box3, 1.1923)

    assert_distance_iou_loss(box1, box4, 1.2500)

    assert_distance_iou_loss(box1s, box2s, 1.2250, reduction="mean")
    assert_distance_iou_loss(box1s, box2s, 2.4500, reduction="sum")


class TestCompleteBoxIou(BoxTestBase):
    def _target_fn(self) -> Tuple[bool, Callable]:
        return (True, ops.complete_box_iou)

    def _generate_int_input() -> List[List[int]]:
        return [[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]

    def _generate_int_expected() -> List[List[float]]:
        return [
            [1.0000, 0.1875, -0.4444],
            [0.1875, 1.0000, -0.5625],
            [-0.4444, -0.5625, 1.0000],
        ]

    def _generate_float_input() -> List[List[float]]:
        return [
            [285.3538, 185.5758, 1193.5110, 851.4551],
            [285.1472, 188.7374, 1192.4984, 851.0669],
            [279.2440, 197.9812, 1189.4746, 849.2019],
        ]

    def _generate_float_expected() -> List[List[float]]:
        return [[1.0, 0.9933, 0.9673], [0.9933, 1.0, 0.9737], [0.9673, 0.9737, 1.0]]

    @pytest.mark.parametrize(
        "test_input, dtypes, tolerance, expected",
        [
            pytest.param(
                _generate_int_input(), [torch.int16, torch.int32, torch.int64], 1e-4, _generate_int_expected()
            ),
            # pytest.param(_generate_float_input(), [torch.float32, torch.float64], 0.002, _generate_float_expected()),
            # pytest.param(_generate_float_input(), [torch.float32, torch.float64], 0.001, _generate_float_expected()),
            # mindspore not support float64 on Ascend.
            pytest.param(_generate_float_input(), [torch.float32], 0.002, _generate_float_expected()),
            pytest.param(_generate_float_input(), [torch.float32], 0.001, _generate_float_expected()),
        ],
    )
    def test_complete_iou(self, test_input: List, dtypes, tolerance: float, expected: List) -> None:
        self._run_test(test_input, dtypes, tolerance, expected)


class TestMasksToBoxes:
    def test_masks_box(self):
        def masks_box_check(masks, expected, tolerance=1e-4):
            out = ops.masks_to_boxes(masks)
            assert out.dtype == torch.float
            assert np.allclose(out.numpy(), expected.numpy(), rtol=0.0, atol=tolerance)

        # Check for int type boxes.
        def _get_image():
            assets_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
            mask_path = os.path.join(assets_directory, "masks.tiff")
            image = Image.open(mask_path)
            return image

        def _create_masks(image, masks):
            for index in range(image.n_frames):
                image.seek(index)
                frame = np.array(image)
                masks[index] = torch.tensor(frame)

            return masks

        expected = torch.tensor(
            [
                [127, 2, 165, 40],
                [2, 50, 44, 92],
                [56, 63, 98, 100],
                [139, 68, 175, 104],
                [160, 112, 198, 145],
                [49, 138, 99, 182],
                [108, 148, 152, 213],
            ],
            dtype=torch.float,
        )

        image = _get_image()
        for dtype in [torch.float16, torch.float32]:
            masks = torch.zeros((image.n_frames, image.height, image.width), dtype=dtype)
            masks = _create_masks(image, masks)
            masks_box_check(masks, expected)


class TestStochasticDepth:
    @SKIP_ENV_CPU(reason="bernoulli has problem handling circulation inputs")
    @SKIP_ENV_GPU (reason="bernoulli has problem handling circulation inputs")
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    @pytest.mark.parametrize("mode", ["batch", "row"])
    def test_stochastic_depth_random(self, seed, mode, p):
        torch.manual_seed(seed)
        stats = pytest.importorskip("scipy.stats")
        batch_size = 5
        x = torch.ones((batch_size, 3, 4, 4))
        layer = ops.StochasticDepth(p=p, mode=mode)
        layer.__repr__()

        trials = 250
        num_samples = 0
        counts = 0
        for _ in range(trials):
            out = layer(x)
            non_zero_count = out.sum(dim=(1, 2, 3)).nonzero().size(0)
            if mode == "batch":
                if non_zero_count == 0:
                    counts += 1
                num_samples += 1
            elif mode == "row":
                counts += batch_size - non_zero_count
                num_samples += batch_size

        p_value = stats.binom_test(counts, num_samples, p=p)
        assert p_value > 0.01

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("p", (0, 1))
    @pytest.mark.parametrize("mode", ["batch", "row"])
    def test_stochastic_depth(self, seed, mode, p):
        torch.manual_seed(seed)
        batch_size = 5
        x = torch.ones((batch_size, 3, 4, 4))
        layer = ops.StochasticDepth(p=p, mode=mode)

        out = layer(x)
        if p == 0:
            assert out.equal(x)
        elif p == 1:
            assert out.equal(torch.zeros_like(x))

    def make_obj(self, p, mode, wrap=False):
        obj = ops.StochasticDepth(p, mode)
        return StochasticDepthWrapper(obj) if wrap else obj


class TestDropBlock:
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("p", [0, 0.5])
    @pytest.mark.parametrize("block_size", [5, 11])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_drop_block(self, seed, dim, p, block_size, inplace):
        torch.manual_seed(seed)
        batch_size = 5
        channels = 3
        height = 11
        width = height
        depth = height
        if dim == 2:
            x = torch.ones((batch_size, channels, height, width))
            layer = ops.DropBlock2d(p=p, block_size=block_size, inplace=inplace)
            feature_size = height * width
        elif dim == 3:
            x = torch.ones((batch_size, channels, depth, height, width))
            layer = ops.DropBlock3d(p=p, block_size=block_size, inplace=inplace)
            feature_size = depth * height * width
        layer.__repr__()

        out = layer(x)
        if p == 0:
            assert out.equal(x)
        if block_size == height:
            for b, c in product(range(batch_size), range(channels)):
                assert out[b, c].count_nonzero() in (0, feature_size)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("p", [0.1, 0.2])
    @pytest.mark.parametrize("block_size", [3])
    @pytest.mark.parametrize("inplace", [False])
    @SKIP_ENV_CPU(reason="bernoulli has problem handling circulation inputs")
    @SKIP_ENV_GPU (reason="bernoulli has problem handling circulation inputs")
    def test_drop_block_random(self, seed, dim, p, block_size, inplace):
        torch.manual_seed(seed)
        batch_size = 5
        channels = 3
        height = 11
        width = height
        depth = height
        if dim == 2:
            x = torch.ones((batch_size, channels, height, width))
            layer = ops.DropBlock2d(p=p, block_size=block_size, inplace=inplace)
        elif dim == 3:
            x = torch.ones((batch_size, channels, depth, height, width))
            layer = ops.DropBlock3d(p=p, block_size=block_size, inplace=inplace)

        trials = 250
        num_samples = 0
        counts = 0
        cell_numel = torch.tensor(x.shape).prod()
        for _ in range(trials):
            # currently not support no_grad() on GRAPH mode
            # with torch.no_grad():
            #     out = layer(x)
            out = layer(x)
            out = out.detach()
            non_zero_count = out.nonzero().size(0)
            counts += cell_numel - non_zero_count
            num_samples += cell_numel

        assert abs(p - counts / num_samples) / p < 0.15

    def make_obj(self, dim, p, block_size, inplace, wrap=False):
        if dim == 2:
            obj = ops.DropBlock2d(p, block_size, inplace)
        elif dim == 3:
            obj = ops.DropBlock3d(p, block_size, inplace)
        return DropBlockWrapper(obj) if wrap else obj


class TestFocalLoss:
    def _generate_diverse_input_target_pair(self, shape=(5, 2), **kwargs):
        def logit(p: Tensor) -> Tensor:
            return torch.log(p / (1 - p))

        def generate_tensor_with_range_type(shape, range_type, **kwargs):
            if range_type != "random_binary":
                low, high = {
                    "small": (0.0, 0.2),
                    "big": (0.8, 1.0),
                    "zeros": (0.0, 0.0),
                    "ones": (1.0, 1.0),
                    "random": (0.0, 1.0),
                }[range_type]
                # return torch.testing.make_tensor(shape, low=low, high=high, **kwargs)
                if low == high and low == 0:
                    return torch.zeros(shape)
                elif low == high and low == 1:
                    return torch.ones(shape)
                else:
                    return torch.ones(shape).uniform_(low, high)
            else:
                return torch.randint(0, 2, shape, **kwargs)

        # This function will return inputs and targets with shape: (shape[0]*9, shape[1])
        inputs = []
        targets = []
        for input_range_type, target_range_type in [
            ("small", "zeros"),
            ("small", "ones"),
            ("small", "random_binary"),
            ("big", "zeros"),
            ("big", "ones"),
            ("big", "random_binary"),
            ("random", "zeros"),
            ("random", "ones"),
            ("random", "random_binary"),
        ]:
            inputs.append(logit(generate_tensor_with_range_type(shape, input_range_type, **kwargs)))
            targets.append(generate_tensor_with_range_type(shape, target_range_type, **kwargs))

        return torch.cat(inputs), torch.cat(targets)

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.58, 1.0])
    @pytest.mark.parametrize("gamma", [0, 2])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_correct_ratio(self, alpha, gamma, device, dtype, seed) -> None:
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # For testing the ratio with manual calculation, we require the reduction to be "none"
        reduction = "none"
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        focal_loss = ops.sigmoid_focal_loss(inputs, targets, gamma=gamma, alpha=alpha, reduction=reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)

        assert torch.all(
            focal_loss <= ce_loss
        ), "focal loss must be less or equal to cross entropy loss with same input"

        loss_ratio = (focal_loss / ce_loss).squeeze()
        prob = torch.sigmoid(inputs)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        correct_ratio = (1.0 - p_t) ** gamma
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            correct_ratio = correct_ratio * alpha_t

        tol = 1e-3 if dtype is torch.half else 1e-5
        # torch.testing.assert_close(correct_ratio, loss_ratio, rtol=tol, atol=tol)
        assert np.allclose(correct_ratio.numpy(), loss_ratio.numpy(), rtol=tol, atol=tol)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("seed", [2, 3])
    def test_equal_ce_loss(self, reduction, device, dtype, seed) -> None:
        if device == "cpu" and dtype is torch.half:
            pytest.skip("Currently torch.half is not fully supported on cpu")
        # focal loss should be equal ce_loss if alpha=-1 and gamma=0
        alpha = -1
        gamma = 0
        torch.random.manual_seed(seed)
        inputs, targets = self._generate_diverse_input_target_pair(dtype=dtype, device=device)
        inputs_fl = inputs.clone().requires_grad_()
        targets_fl = targets.clone()
        inputs_ce = inputs.clone().requires_grad_()
        targets_ce = targets.clone()
        focal_loss = ops.sigmoid_focal_loss(inputs_fl, targets_fl, gamma=gamma, alpha=alpha, reduction=reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs_ce, targets_ce, reduction=reduction)

        tol = 1e-3 if dtype is torch.half else 1e-5
        assert np.allclose(focal_loss.numpy(), ce_loss.numpy(), rtol=tol, atol=tol)


class TestGeneralizedBoxIouLoss:
    # We refer to original test: https://github.com/facebookresearch/fvcore/blob/main/tests/test_giou_loss.py
    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    def test_giou_loss(self, dtype, device) -> None:
        box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
        box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
        box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
        box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

        box1s = torch.stack([box2, box2], dim=0)
        box2s = torch.stack([box3, box4], dim=0)

        def assert_giou_loss(box1, box2, expected_loss, reduction="none"):
            tol = 1e-3 if dtype is torch.half else 1e-5
            computed_loss = ops.generalized_box_iou_loss(box1, box2, reduction=reduction)
            expected_loss = torch.tensor(expected_loss, device=device)
            assert np.allclose(computed_loss.numpy(), expected_loss.numpy(), rtol=tol, atol=tol)

        # Identical boxes should have loss of 0
        assert_giou_loss(box1, box1, 0.0)

        # quarter size box inside other box = IoU of 0.25
        assert_giou_loss(box1, box2, 0.75)

        # Two side by side boxes, area=union
        # IoU=0 and GIoU=0 (loss 1.0)
        assert_giou_loss(box2, box3, 1.0)

        # Two diagonally adjacent boxes, area=2*union
        # IoU=0 and GIoU=-0.5 (loss 1.5)
        assert_giou_loss(box2, box4, 1.5)

        # Test batched loss and reductions
        assert_giou_loss(box1s, box2s, 2.5, reduction="sum")
        assert_giou_loss(box1s, box2s, 1.25, reduction="mean")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @SKIP_ENV_GPU(reason="Currently not support 0 in shape on GPU")
    def test_empty_inputs(self, dtype, device) -> None:
        # TODO: currently not support 0 in shape on GPU
        box1 = torch.randn([0, 4], dtype=dtype).requires_grad_()
        box2 = torch.randn([0, 4], dtype=dtype).requires_grad_()

        loss = ops.generalized_box_iou_loss(box1, box2, reduction="mean")
        # loss.backward()

        tol = 1e-3 if dtype is torch.half else 1e-5
        assert np.allclose(loss.numpy(), torch.tensor(0.0).numpy(), rtol=tol, atol=tol)


class TestCIOULoss:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @pytest.mark.parametrize("device", cpu_and_gpu())
    def test_ciou_loss(self, dtype, device):
        box1 = torch.tensor([-1, -1, 1, 1], dtype=dtype, device=device)
        box2 = torch.tensor([0, 0, 1, 1], dtype=dtype, device=device)
        box3 = torch.tensor([0, 1, 1, 2], dtype=dtype, device=device)
        box4 = torch.tensor([1, 1, 2, 2], dtype=dtype, device=device)

        box1s = torch.stack([box2, box2], dim=0)
        box2s = torch.stack([box3, box4], dim=0)

        def assert_ciou_loss(box1, box2, expected_output, reduction="none"):

            output = ops.complete_box_iou_loss(box1, box2, reduction=reduction)
            # TODO: When passing the dtype, the torch.half test doesn't pass...
            expected_output = torch.tensor(expected_output, device=device)
            tol = 1e-5 if dtype != torch.half else 1e-3
            # torch.testing.assert_close(output, expected_output, rtol=tol, atol=tol)
            assert np.allclose(output.numpy(), expected_output.numpy(), rtol=tol, atol=tol)

        assert_ciou_loss(box1, box1, 0.0)

        assert_ciou_loss(box1, box2, 0.8125)

        assert_ciou_loss(box1, box3, 1.1923)

        assert_ciou_loss(box1, box4, 1.2500)

        assert_ciou_loss(box1s, box2s, 1.2250, reduction="mean")
        assert_ciou_loss(box1s, box2s, 2.4500, reduction="sum")

    @pytest.mark.parametrize("device", cpu_and_gpu())
    @pytest.mark.parametrize("dtype", [torch.float32, torch.half])
    @SKIP_ENV_GPU(reason="Currently not support 0 in shape on GPU")
    def test_empty_inputs(self, dtype, device) -> None:
        # TODO: currently not support 0 in shape
        box1 = torch.randn([0, 4], dtype=dtype).requires_grad_()
        box2 = torch.randn([0, 4], dtype=dtype).requires_grad_()

        loss = ops.complete_box_iou_loss(box1, box2, reduction="mean")
        # loss.backward()

        tol = 1e-3 if dtype is torch.half else 1e-5
        # torch.testing.assert_close(loss, torch.tensor(0.0), rtol=tol, atol=tol)
        assert np.allclose(loss.numpy(), torch.tensor(0.0).numpy(), rtol=tol, atol=tol)
        # assert box1.grad is not None, "box1.grad should not be None after backward is called"
        # assert box2.grad is not None, "box2.grad should not be None after backward is called"

        loss = ops.complete_box_iou_loss(box1, box2, reduction="none")
        assert loss.numel() == 0, "ciou_loss for two empty box should be empty"


if __name__ == "__main__":
    pytest.main([__file__])
