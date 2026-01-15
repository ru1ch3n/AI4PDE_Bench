import numpy as np

from ai4pde_bench.masks import MaskConfig, make_mask


def test_mask_shapes_and_ratio():
    shape = (32, 32)
    cfgs = [
        MaskConfig(name="m1_random", obs_ratio=0.1, seed=0),
        MaskConfig(name="m2_regular", obs_ratio=0.1, seed=0),
        MaskConfig(name="m3_block", obs_ratio=0.1, seed=0),
    ]
    for cfg in cfgs:
        m = make_mask(shape, cfg)
        assert m.shape == shape
        assert m.dtype == np.float32
        r = m.mean()
        assert 0.05 <= r <= 0.15  # loose
