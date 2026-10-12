# Why the current baseline does not naively scale to 384^3

This is a characterization of the present implementation, not a requirement to solve full-volume generation in the first baseline.

With isotropic `4^3` patches, a `64^3` crop produces `16^3 = 4,096` tokens at the highest pyramid stage. A `384^3` field produces `96^3 = 884,736` tokens. Dense self-attention stores or computes interactions proportional to the square of token count:

| Volume | Patch grid | Tokens | Pairwise token interactions |
|---|---:|---:|---:|
| `16^3` | `4^3` | 64 | 4,096 |
| `32^3` | `8^3` | 512 | 262,144 |
| `64^3` | `16^3` | 4,096 | 16,777,216 |
| `384^3` | `96^3` | 884,736 | 782,757,789,696 |

The token count grows by 216 times from `64^3` to `384^3`; the dense-attention interaction count grows by 46,656 times. The raw three-channel field is only about 680 MiB in float32, but attention—not storage of the input field—is the dominant obstacle.

Candidate changes to study later include larger or hierarchical patches, a learned 3-D latent codec, local/windowed or sparse attention, domain decomposition with overlap, and patch-based diffusion/flow methods. SpotDiffusion is relevant as a comparison family because it treats large-domain generation through patches rather than applying global attention to every full-resolution token. Any such change affects boundary consistency and long-range turbulent structure, so it should be treated as a separate research question rather than silently folded into this baseline.
