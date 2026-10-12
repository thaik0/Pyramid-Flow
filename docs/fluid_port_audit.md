# Fluid-volume port audit

Upstream revision audited: `a012faa1dc4d71301a7a153c7f9554c081947ea2`.

## Execution maps

Training:

```text
scripts/train_pyramid_flow*.sh
  -> train/train_pyramid_flow.py:get_args/main
  -> argparse flags (there is no file-based training config)
  -> LengthGroupedVideoTextDataset + TemporalLengthBucketeer
  -> video tensor [B,C,T,H,W]
     B=batch, C=VAE latent channels (16), T=latent video frames,
     H/W=latent image height/width
  -> PyramidDiTForVideoGeneration.__call__/get_vae_latent
  -> get_pyramid_latent (H/W-only pyramid)
  -> add_pyramid_noise[_with_temporal_pyramid]
  -> PyramidFluxTransformer or PyramidDiffusionMMDiT
  -> MSE(model velocity, start_point - end_point)
  -> Accelerator.save_state(checkpoint-<global_step>)
```

Sampling:

```text
inference_multigpu.py
  -> PyramidDiTForVideoGeneration.generate
  -> Gaussian noise [B,C,T,H/8,W/8]
  -> H/W-only low-resolution start
  -> generate_one_unit, once per autoregressive temporal unit
  -> for each pyramid scale: Euler steps + DiT evaluations
  -> nearest-neighbor H/W lift + correlated corrective block noise
  -> concatenate temporal units -> causal video VAE decode -> frames
```

## Shapes and implementation assumptions

- The video dataset loads precomputed VAE latents shaped `[B,16,T,H,W]`. `T` is explicitly ordered video time.
- MMDiT's misleadingly named `PatchEmbed3D` uses `Conv2d` on each frame independently. It forms `2x2` patches in `H,W`; it does not patch `T`.
- Flux likewise creates tokens at every `(T,H,W)` location and assigns three-axis RoPE coordinates. Its configured RoPE allocation is `[16,24,24]`, making the first axis distinct from the two image axes.
- MMDiT has 2-D spatial position embeddings plus separate temporal sinusoidal/RoPE embeddings.
- With `use_temporal_causal`, attention masks prevent a token from attending to future frames.
- Temporal-pyramid training and generation are autoregressive: earlier generated frames are conditions for later units. `frame_per_unit`, ordered time indices, and history corruption are video semantics.
- The causal video VAE uses video-specific temporal compression and first-frame versus later-frame scaling.
- The spatial pyramid uses `rearrange(... '(b t) c h w')` followed by 2-D interpolation. Only `H,W` change across stages; `T` never changes.

The DiT output has the same semantic shape as its input: `[B,C,T,H,W]`. MMDiT unpatchifies `2x2` image patches; Flux packs each `2x2` patch into channels before its linear projection and reverses that operation at output.

## Can `T` simply become fluid depth `D`?

Not as a faithful, practical 3-D baseline.

Disabling causal attention and temporal autoregression would remove the strongest semantic violation, and global self-attention could then exchange information across slices. However, the representation would remain anisotropic: depth is unpatched, separately position-encoded, and excluded from every pyramid resize and scale transition. At `384^3`, a `2x2` H/W patcher creates `384 * 192 * 192 = 14,155,776` data tokens per sample before text tokens, which is not computationally viable for dense attention. The released 16-channel VAE/DiT weights also cannot directly accept three raw velocity channels.

## Component classification

| Component | Class | Reason |
|---|---|---|
| Rectified-flow interpolation and target | A | Tensor-rank agnostic; target is `start_point - end_point`. |
| Per-stage scheduler and timestep ranges | A | Independent of video content. |
| MSE flow loss | A | Tensor-rank agnostic. |
| Checkpoint/seed plumbing | A | Accelerator state plus `set_seed(seed, device_specific=True)`. |
| Full-token transformer blocks | B | Usable with 3-D tokens and noncausal masks. |
| Patch/embed and unpatchify | B | Must patch all `D,H,W`, accept 3 channels, and provide 3-D positions. |
| Spatial pyramid and lifting | B | Must resize `D,H,W`, not just `H,W`; corrective noise must become 3-D blocks. |
| Text conditioning | C | No conditioning variable was specified for unconditional fluid generation. |
| Causal video VAE | C | Video/RGB-specific, temporally causal, wrong channels and scaling. |
| Temporal causal mask | C | Depth has no past/future direction. |
| Temporal AR pyramid/history corruption | C | Generates ordered future frames rather than one simultaneous volume. |
| Patch size / crop size / latent codec choice | D | Determines feasibility and whether pretrained weights can be reused; requires mentor-approved experimental policy. |
| Train/validation/test split | D | Files are one correlated trajectory; no split policy was supplied. |
| Normalization | D pending statistics | Must be selected only after scanning real values. |

## Exact pyramid behavior

The default three stages are scale factors `[1,2,4]` with nominal global sigma boundaries `[0,1/3,2/3,1]`. Training constructs high-resolution Gaussian noise, repeatedly downsamples `H,W` by two with bilinear interpolation and multiplies the result by two. For stage zero, the path begins at scale-matched noise. Later stages begin from a nearest-neighbor upsampled previous clean scale mixed with noise at that stage's corrected start sigma. Non-final stages end at a mixture of their clean target and noise; the final stage ends at the clean target. The scheduler remaps each stage's path length to its own 1,000 training ratios.

At inference, each stage runs Euler integration over its assigned interval. Between stages, the sample is lifted by nearest-neighbor interpolation in `H,W`. Correlated `2x2` block noise is then injected using `gamma=1/3` to reduce block artifacts. None of these operations changes `T`.

## Safe adaptation completed

`dataset/fluid_volume_dataset.py` is a lazy, traceable adapter. It:

- opens only the selected NetCDF file and timestep;
- returns `volume: float32 [C,D,H,W]`, with `C=(u,v,w)`;
- returns filename, absolute path, and timestep;
- sorts numeric filenames correctly;
- drops timestep zero only across consecutive numbered files;
- applies no normalization by default;
- optionally applies reversible per-channel standardization from an explicit JSON file.

`tools/scan_fluid_dataset.py` samples a bounded, seeded set of indexed timesteps and records the selected file/timestep pairs along with per-channel mean, standard deviation, minimum, and maximum. This is the prerequisite for proposing normalization.

## Deliberately not changed

The provisional baseline is now implemented as an explicit, opt-in path rather than changing video defaults. It uses deterministic `64^3` crops, `4^3` patches, symmetric 3-D RoPE, noncausal attention, dummy unconditional context, and isotropic `D,H,W` pyramid/lifting. It keeps the upstream MMDiT blocks, scheduler, stage ranges, flow target, MSE loss, AdamW settings, and Euler update. The released video checkpoint is not loaded because its 16-channel 2-D input/output layers are shape-incompatible.

`train/train_fluid_pyramid_flow.py` defaults to one optimization step and saves `config.json`, `metrics.jsonl`, the seed, traceability metadata, RNG states, model state, and optimizer state. It is intentionally not a full training launch.
