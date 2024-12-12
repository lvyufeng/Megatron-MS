import random

import numpy as np
import pytest

def pytest_configure(config):
    # register an additional marker (see pytest_collection_modifyitems)
    config.addinivalue_line("markers", "needs_cuda: mark for tests that rely on a CUDA device")
    config.addinivalue_line("markers", "dont_collect: mark for tests that should not be collected")