import os
import numpy as np
from mindspore import context
import mindspore as ms

import torch as torch
from torchvision.models import alexnet

context.set_context(mode=ms.PYNATIVE_MODE)
import pytest
@pytest.mark.skip("Currently, saving model structure (modules) is not supported.")
def test_save_load():
    # Create test data
    x = np.random.rand(1, 3, 224, 224)
    x = x.astype(np.float32)
    inputs = torch.tensor(x)

    # Save model
    model = alexnet(pretrained=False)
    torch.save(model, "alexnet.pth")
    # new_model = torch.load('alexnet.ckpt')
    # out = new_model(inputs)
    # print(out.shape)

def test_save_load_statedict():
    # Create test data
    x_fixed = np.ones((1, 3, 224, 224))
    x_fixed = x_fixed.astype(np.float32)
    fixed_inputs = torch.tensor(x_fixed)

    model = alexnet(pretrained=False)
    model.eval()
    output = model(fixed_inputs)
    # Save state dict
    state_dict = model.state_dict()
    torch.save(state_dict, 'alexnet_dict.ckpt')
    # Loading state dict
    new_model = alexnet(pretrained=False)
    parm = torch.load('alexnet_dict.ckpt')
    new_model.load_state_dict(parm)
    new_model.eval()
    output_load = new_model(fixed_inputs)
    assert np.allclose(output.detach().numpy(), output_load.detach().numpy())


def test_save_load_statedict_with_dict():
    # Create test data
    x_fixed = np.ones((1, 3, 224, 224))
    x_fixed = x_fixed.astype(np.float32)
    fixed_inputs = torch.tensor(x_fixed)

    model = alexnet(pretrained=False)
    model.eval()
    output = model(fixed_inputs)
    # Save state dict
    _state_dict = model.state_dict()
    state_dict = {'state_dict': _state_dict}
    torch.save(state_dict, 'alexnet_dict.pth')
    # Loading state dict
    new_model = alexnet(pretrained=False)
    parm = torch.load('alexnet_dict.pth')
    new_model.load_state_dict(parm['state_dict'])
    new_model.eval()
    output_load = new_model(fixed_inputs)
    assert np.allclose(output.detach().numpy(), output_load.detach().numpy())

if __name__ == '__main__':
    test_save_load()
    test_save_load_statedict()
    test_save_load_statedict_with_dict()
