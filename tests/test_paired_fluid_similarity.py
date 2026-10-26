import math

import torch

from tools.evaluate_paired_fluid_similarity import psnr_3d, ssim_3d


def test_identical_fields_have_perfect_similarity():
    field = torch.randn(1, 3, 8, 8, 8)
    assert math.isinf(psnr_3d(field, field, 6.0))
    assert abs(ssim_3d(field, field, 6.0) - 1.0) < 1e-6


def test_psnr_matches_known_constant_error():
    target = torch.zeros(1, 1, 4, 4, 4)
    prediction = torch.ones_like(target)
    assert abs(psnr_3d(prediction, target, 2.0) - 6.0206) < 1e-3
