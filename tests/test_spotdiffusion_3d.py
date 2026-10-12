import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "pyramid_dit" / "spotdiffusion_3d.py"
SPEC = importlib.util.spec_from_file_location("spotdiffusion_3d", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _random_full_volume(seed=17):
    generator = torch.Generator().manual_seed(seed)
    # int8 keeps the real 384^3 geometry below one GiB during round-trip tests
    # and permits exact equality rather than a floating-point tolerance.
    return torch.randint(
        -128,
        128,
        (1, 3, 384, 384, 384),
        dtype=torch.int8,
        generator=generator,
    )


def test_fixed_tiling_round_trip_at_target_shape():
    volume = _random_full_volume()
    patches, spec = MODULE.split_nonoverlapping_3d(volume, 64)

    assert spec.grid_shape == (6, 6, 6)
    assert spec.patches_per_sample == 216
    assert patches.shape == (216, 3, 64, 64, 64)
    reconstructed = MODULE.stitch_nonoverlapping_3d(patches, spec)
    assert torch.equal(reconstructed, volume)


def test_shift_split_stitch_inverse_shift_round_trip():
    volume = _random_full_volume(seed=29)
    shift = MODULE.sample_shift_schedule(1, 64, seed=2024)[0]
    assert shift != (0, 0, 0)

    shifted = MODULE.cyclic_shift_3d(volume, shift)
    patches, spec = MODULE.split_nonoverlapping_3d(shifted, 64)
    stitched = MODULE.stitch_nonoverlapping_3d(patches, spec)
    reconstructed = MODULE.inverse_cyclic_shift_3d(stitched, shift)

    assert torch.equal(reconstructed, volume)


def test_patch_coordinates_cover_every_voxel_exactly_once():
    origins = MODULE.patch_origins_3d((384, 384, 384), 64)
    assert len(origins) == 216
    assert len(set(origins)) == 216

    coverage = torch.zeros((384, 384, 384), dtype=torch.uint8)
    for depth, height, width in origins:
        coverage[
            depth : depth + 64,
            height : height + 64,
            width : width + 64,
        ] += 1

    assert torch.all(coverage == 1)


def test_shift_schedule_is_seeded_and_samples_each_axis():
    first = MODULE.sample_shift_schedule(30, 64, seed=42)
    repeated = MODULE.sample_shift_schedule(30, 64, seed=42)
    different = MODULE.sample_shift_schedule(30, 64, seed=43)

    assert first == repeated
    assert first != different
    assert len(set(first)) > 1
    assert all(len(shift) == 3 for shift in first)
    assert all(0 <= coordinate < 64 for shift in first for coordinate in shift)


def test_channel_axis_is_neither_shifted_nor_patched():
    volume = torch.arange(3).reshape(1, 3, 1, 1, 1).expand(1, 3, 8, 8, 8)
    shifted = MODULE.cyclic_shift_3d(volume, (1, 2, 3))
    patches, spec = MODULE.split_nonoverlapping_3d(shifted, 4)

    assert patches.shape == (8, 3, 4, 4, 4)
    for channel in range(3):
        assert torch.all(patches[:, channel] == channel)
    assert torch.equal(MODULE.stitch_nonoverlapping_3d(patches, spec), shifted)
