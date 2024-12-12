import os
import re
import sys
import tempfile
from io import BytesIO

import numpy as np
import pytest
import torch as torch
import torchvision.transforms.functional as F
import torchvision.utils as utils
from PIL import __version__ as PILLOW_VERSION, Image, ImageColor


PILLOW_VERSION = tuple(int(x) for x in PILLOW_VERSION.split("."))

boxes = torch.tensor([[0, 0, 20, 20], [0, 0, 0, 0], [10, 15, 30, 35], [23, 35, 93, 95]], dtype=torch.float)

keypoints = torch.tensor([[[10, 10], [5, 5], [2, 2]], [[20, 20], [30, 30], [3, 3]]], dtype=torch.float)


def test_make_grid_not_inplace():
    t = torch.rand(5, 3, 10, 10)
    t_clone = t.clone()

    utils.make_grid(t, normalize=False)
    assert np.allclose(t.numpy(), t_clone.numpy())

    utils.make_grid(t, normalize=True, scale_each=False)
    assert np.allclose(t.numpy(), t_clone.numpy())

    utils.make_grid(t, normalize=True, scale_each=True)
    assert np.allclose(t.numpy(), t_clone.numpy())


def test_normalize_in_make_grid():
    t = torch.rand(5, 3, 10, 10) * 255
    norm_max = torch.tensor(1.0)
    norm_min = torch.tensor(0.0)

    grid = utils.make_grid(t, normalize=True)
    grid_max = torch.max(grid)
    grid_min = torch.min(grid)

    # Rounding the result to one decimal for comparison
    n_digits = 1
    rounded_grid_max = torch.round(grid_max * 10**n_digits) / (10**n_digits)
    rounded_grid_min = torch.round(grid_min * 10**n_digits) / (10**n_digits)
    assert np.allclose(norm_max.numpy(), rounded_grid_max.numpy())
    assert np.allclose(norm_min.numpy(), rounded_grid_min.numpy())


@pytest.mark.skipif(sys.platform in ("win32", "cygwin"), reason="temporarily disabled on Windows")
def test_save_image():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        t = torch.rand(2, 3, 64, 64)
        utils.save_image(t, f.name)
        assert os.path.exists(f.name), "The image is not present after save"


@pytest.mark.skipif(sys.platform in ("win32", "cygwin"), reason="temporarily disabled on Windows")
def test_save_image_single_pixel():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        t = torch.rand(1, 3, 1, 1)
        utils.save_image(t, f.name)
        assert os.path.exists(f.name), "The pixel image is not present after save"


@pytest.mark.skipif(sys.platform in ("win32", "cygwin"), reason="temporarily disabled on Windows")
def test_save_image_file_object():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        t = torch.rand(2, 3, 64, 64)
        utils.save_image(t, f.name)
        img_orig = Image.open(f.name)
        fp = BytesIO()
        utils.save_image(t, fp, format="png")
        img_bytes = Image.open(fp)
        assert np.allclose(F.pil_to_tensor(img_orig).numpy(), F.pil_to_tensor(img_bytes).numpy())


@pytest.mark.skipif(sys.platform in ("win32", "cygwin"), reason="temporarily disabled on Windows")
def test_save_image_single_pixel_file_object():
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        t = torch.rand(1, 3, 1, 1)
        utils.save_image(t, f.name)
        img_orig = Image.open(f.name)
        fp = BytesIO()
        utils.save_image(t, fp, format="png")
        img_bytes = Image.open(fp)
        assert np.allclose(F.pil_to_tensor(img_orig).numpy(), F.pil_to_tensor(img_bytes).numpy())


def test_draw_boxes():
    img = torch.full((3, 100, 100), 255, dtype=torch.uint8)
    img_cp = img.clone()
    boxes_cp = boxes.clone()
    labels = ["a", "b", "c", "d"]
    colors = ["green", "#FF00FF", (0, 255, 0), "red"]
    result = utils.draw_bounding_boxes(img, boxes, labels=labels, colors=colors, fill=True)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "draw_boxes_util.png")
    if not os.path.exists(path):
        res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
        res.save(path)

    if PILLOW_VERSION >= (8, 2):
        # The reference image is only valid for new PIL versions
        expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
        assert np.allclose(result.numpy(), expected.numpy())

    # Check if modification is not in place
    assert np.allclose(boxes.numpy(), boxes_cp.numpy())
    assert np.allclose(img.numpy(), img_cp.numpy())


@pytest.mark.parametrize("colors", [None, ["red", "blue", "#FF00FF", (1, 34, 122)], "red", "#FF00FF", (1, 34, 122)])
def test_draw_boxes_colors(colors):
    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
    utils.draw_bounding_boxes(img, boxes, fill=False, width=7, colors=colors)


def test_draw_boxes_vanilla():
    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
    img_cp = img.clone()
    boxes_cp = boxes.clone()
    result = utils.draw_bounding_boxes(img, boxes, fill=False, width=7, colors="white")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "draw_boxes_vanilla.png")
    if not os.path.exists(path):
        res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
        res.save(path)

    expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
    assert np.allclose(result.numpy(), expected.numpy())
    # Check if modification is not in place
    assert np.allclose(boxes.numpy(), boxes_cp.numpy())
    assert np.allclose(img.numpy(), img_cp.numpy())


def test_draw_boxes_grayscale():
    img = torch.full((1, 4, 4), fill_value=255, dtype=torch.uint8)
    boxes = torch.tensor([[0, 0, 3, 3]], dtype=torch.int64)
    bboxed_img = utils.draw_bounding_boxes(image=img, boxes=boxes, colors=["#1BBC9B"])
    assert bboxed_img.size(0) == 3


def test_draw_boxes_warning():
    img = torch.full((3, 100, 100), 255, dtype=torch.uint8)

    with pytest.warns(UserWarning, match=re.escape("Argument 'font_size' will be ignored since 'font' is not set.")):
        utils.draw_bounding_boxes(img, boxes, font_size=11)


def test_draw_no_boxes():
    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
    boxes = torch.full((0, 4), 0, dtype=torch.float)
    with pytest.warns(UserWarning, match=re.escape("boxes doesn't contain any box. No box was drawn")):
        res = utils.draw_bounding_boxes(img, boxes)
        # Check that the function didnt change the image
        assert res.eq(img).all()


@pytest.mark.parametrize(
    "colors",
    [
        None,
        "blue",
        "#FF00FF",
        (1, 34, 122),
        ["red", "blue"],
        ["#FF00FF", (1, 34, 122)],
    ],
)
@pytest.mark.parametrize("alpha", (0, 0.5, 0.7, 1))
def test_draw_segmentation_masks(colors, alpha):
    """This test makes sure that masks draw their corresponding color where they should"""
    num_masks, h, w = 2, 100, 100
    dtype = torch.uint8
    img = torch.randint(0, 256, size=(3, h, w), dtype=torch.int8)
    img = torch.tensor(img, dtype = torch.uint8)
    masks = torch.randint(0, 2, (num_masks, h, w), dtype=torch.bool)

    # For testing we enforce that there's no overlap between the masks. The
    # current behaviour is that the last mask's color will take priority when
    # masks overlap, but this makes testing slightly harder so we don't really
    # care
    overlap = masks[0] & masks[1]
    masks[:, overlap] = False

    out = utils.draw_segmentation_masks(img, masks, colors=colors, alpha=alpha)
    assert out.dtype == dtype
    assert out is not img

    # Make sure the image didn't change where there's no mask
    masked_pixels = masks[0] | masks[1]
    assert np.allclose(img[:, ~masked_pixels].numpy(), out[:, ~masked_pixels].numpy())

    if colors is None:
        colors = utils._generate_color_palette(num_masks)
    elif isinstance(colors, str) or isinstance(colors, tuple):
        colors = [colors]

    # Make sure each mask draws with its own color
    for mask, color in zip(masks, colors):
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        color = torch.tensor(color, dtype=dtype)

        if alpha == 1:
            assert (out[:, mask] == color[:, None]).all()
        elif alpha == 0:
            assert (out[:, mask] == img[:, mask]).all()

        interpolated_color = (img[:, mask] * (1 - alpha) + color[:, None] * alpha).to(dtype)
        assert np.allclose(out[:, mask].numpy(), interpolated_color.numpy(), rtol=0.0, atol=1.0)


def test_draw_segmentation_masks_errors():
    h, w = 10, 10

    img = torch.randint(0, 256, size=(3, h, w), dtype=torch.int8)
    img = torch.tensor(img, dtype = torch.uint8)
    masks = torch.randint(0, 2, (3, h, w), dtype=torch.int8)
    masks = torch.tensor(masks, dtype=torch.bool)

    with pytest.raises(TypeError, match="The image must be a tensor"):
        utils.draw_segmentation_masks(image="Not A Tensor Image", masks=masks)
    with pytest.raises(ValueError, match="The image dtype must be"):
        img_bad_dtype = torch.randint(0, 256, size=(3, h, w), dtype=torch.int64)
        utils.draw_segmentation_masks(image=img_bad_dtype, masks=masks)
    with pytest.raises(ValueError, match="Pass individual images, not batches"):
        batch = torch.randint(0, 256, size=(10, 3, h, w), dtype=torch.int8)
        batch = batch.to(torch.uint8)
        utils.draw_segmentation_masks(image=batch, masks=masks)
    with pytest.raises(ValueError, match="Pass an RGB image"):
        one_channel = torch.randint(0, 256, size=(1, h, w), dtype=torch.int8)
        one_channel = one_channel.to(torch.uint8)
        utils.draw_segmentation_masks(image=one_channel, masks=masks)
    with pytest.raises(ValueError, match="The masks must be of dtype bool"):
        masks_bad_dtype = torch.randint(0, 2, size=(h, w), dtype=torch.int8).to(torch.float32)
        utils.draw_segmentation_masks(image=img, masks=masks_bad_dtype)
    with pytest.raises(ValueError, match="masks must be of shape"):
        masks_bad_shape = torch.randint(0, 2, size=(3, 2, h, w), dtype=torch.int8).to(torch.bool)
        utils.draw_segmentation_masks(image=img, masks=masks_bad_shape)
    with pytest.raises(ValueError, match="must have the same height and width"):
        masks_bad_shape = torch.randint(0, 2, size=(h + 4, w), dtype=torch.int8).to(torch.bool)
        utils.draw_segmentation_masks(image=img, masks=masks_bad_shape)
    with pytest.raises(ValueError, match="There are more masks"):
        utils.draw_segmentation_masks(image=img, masks=masks, colors=[])



def test_draw_no_segmention_mask():
    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
    masks = torch.full((0, 100, 100), 0, dtype=torch.bool)
    with pytest.warns(UserWarning, match=re.escape("masks doesn't contain any mask. No mask was drawn")):
        res = utils.draw_segmentation_masks(img, masks)
        # Check that the function didnt change the image
        assert res.eq(img).all()


def test_draw_keypoints_vanilla():
    # Keypoints is declared on top as global variable
    keypoints_cp = keypoints.clone()

    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
    img_cp = img.clone()
    result = utils.draw_keypoints(
        img,
        keypoints,
        colors="red",
        connectivity=[
            (0, 1),
        ],
    )
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "draw_keypoint_vanilla.png")
    if not os.path.exists(path):
        res = Image.fromarray(result.permute(1, 2, 0).contiguous().numpy())
        res.save(path)

    expected = torch.as_tensor(np.array(Image.open(path))).permute(2, 0, 1)
    assert np.allclose(result.numpy(), expected.numpy())
    # Check that keypoints are not modified inplace
    assert np.allclose(keypoints.numpy(), keypoints_cp.numpy())
    # Check that image is not modified in place
    assert np.allclose(img.numpy(), img_cp.numpy())


@pytest.mark.parametrize("colors", ["red", "#FF00FF", (1, 34, 122)])
def test_draw_keypoints_colored(colors):
    # Keypoints is declared on top as global variable
    keypoints_cp = keypoints.clone()

    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)
    img_cp = img.clone()
    result = utils.draw_keypoints(
        img,
        keypoints,
        colors=colors,
        connectivity=[
            (0, 1),
        ],
    )
    assert result.size(0) == 3
    assert np.allclose(keypoints.numpy(), keypoints_cp.numpy())
    assert np.allclose(img.numpy(), img_cp.numpy())


def test_draw_keypoints_errors():
    h, w = 10, 10
    img = torch.full((3, 100, 100), 0, dtype=torch.uint8)

    with pytest.raises(TypeError, match="The image must be a tensor"):
        utils.draw_keypoints(image="Not A Tensor Image", keypoints=keypoints)
    with pytest.raises(ValueError, match="The image dtype must be"):
        img_bad_dtype = torch.full((3, h, w), 0, dtype=torch.int64)
        utils.draw_keypoints(image=img_bad_dtype, keypoints=keypoints)
    with pytest.raises(ValueError, match="Pass individual images, not batches"):
        batch = torch.randint(0, 256, size=(10, 3, h, w), dtype=torch.int8).to(torch.uint8)
        utils.draw_keypoints(image=batch, keypoints=keypoints)
    with pytest.raises(ValueError, match="Pass an RGB image"):
        one_channel = torch.randint(0, 256, size=(1, h, w), dtype=torch.int8).to(torch.uint8)
        utils.draw_keypoints(image=one_channel, keypoints=keypoints)
    with pytest.raises(ValueError, match="keypoints must be of shape"):
        invalid_keypoints = torch.tensor([[10, 10, 10, 10], [5, 6, 7, 8]], dtype=torch.float)
        utils.draw_keypoints(image=img, keypoints=invalid_keypoints)



@pytest.mark.parametrize(
    "input_flow, match",
    (
        (torch.full((3, 10, 10), 0, dtype=torch.float), "Input flow should have shape"),
        (torch.full((5, 3, 10, 10), 0, dtype=torch.float), "Input flow should have shape"),
        (torch.full((2, 10), 0, dtype=torch.float), "Input flow should have shape"),
        (torch.full((5, 2, 10), 0, dtype=torch.float), "Input flow should have shape"),
        (torch.full((2, 10, 30), 0, dtype=torch.int), "Flow should be of dtype torch.float"),
    ),
)
def test_flow_to_image_errors(input_flow, match):
    with pytest.raises(ValueError, match=match):
        utils.flow_to_image(flow=input_flow)


if __name__ == "__main__":
    pytest.main([__file__])
