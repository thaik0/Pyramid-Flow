from .flux_modules import FluxSingleTransformerBlock, FluxTransformerBlock, FluxTextEncoderWithMask
from .mmdit_modules import JointTransformerBlock, SD3TextEncoderWithMask


def __getattr__(name):
    """Avoid importing the video VAE when only the fluid DiT is requested."""
    if name == "PyramidDiTForVideoGeneration":
        from .pyramid_dit_for_video_gen_pipeline import PyramidDiTForVideoGeneration
        return PyramidDiTForVideoGeneration
    raise AttributeError(name)
