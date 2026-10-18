[← Repository](../README.md) · [Documentation](README.md)

# Original diffusion interfaces

The original fMRI and Hb diffusion architectures retain the author implementations, with separate encoded inputs, target coordinates and fitted weights. Use this page with the [prepared data formats](data.md#original-diffusion-generators) and [model correspondence](methods.md#original-diffusion-models).

[Architectures](#architectures) · [Training](#training) · [Inference](#inference) · [Target preparation](#target-preparation) · [Checkpoints](#checkpoints-and-software-configurations)

## Architectures

| Branch | Target / initial noise | EEG condition | Denoiser output¹ | Production backbone |
| --- | --- | --- | --- | --- |
| [fMRI DiT](../src/unifiedneurogen/dit.py) | `[batch, 1, 2560]` | `[batch, 2560, 1200]` | `[batch, 2, 2560]` | 24 blocks, width 1200, 12 heads, patch 4 |
| [Hb DiT](../src/unifiedneurogen/gdit.py) | `[batch, 144, 128]` | `[batch, 144, 1000]` | `[batch, 288, 128]` | 28 blocks, width 1000, 10 heads, patch 1 |

¹ The supplied configurations use `learn_sigma=true`: the output channel halves predict noise and learned diffusion variance. Setting `learn_sigma=false` selects fixed variance and keeps the denoiser output channel count equal to the target channel count.

The factories validate the declared condition width because the original `Adaptor` preserves its last feature dimension. Diffusion timesteps are separate from physiological time. These models do not contain M10's learned history banks or the linear models' fixed double-gamma history candidate.

## Training

Supply your own compatible prepared arrays. The fMRI branch uses [dit_fmri.json](../configs/dit_fmri.json):

```bash
python -m unifiedneurogen dit-train \
  --data paired_encodings.npz --config configs/dit_fmri.json \
  --device cuda --output outputs/dit_training
```

For Hb, add `--branch hb` and use [dit_hb.json](../configs/dit_hb.json). NPZ files contain `eeg`, `target` and participant-level `split` arrays; their shapes are listed in [data formats](data.md#original-diffusion-generators).

Both branches use a 1,000-step linear DDPM schedule. With the supplied `learn_sigma=true` settings, the objective combines noise MSE with the original variational-bound loss. The CLI also supports fixed variance through `learn_sigma=false` in the model configuration.

## Inference

Sampling uses ordinary conditional `forward` with DDIM and `eta=0`:

```bash
python -m unifiedneurogen dit-sample \
  --eeg eeg_encodings.npy --checkpoint outputs/dit_training/dit.pt \
  --config configs/dit_fmri.json --steps 50 --device cuda \
  --output outputs/dit_generated.npz
```

Again, add `--branch hb` and use `configs/dit_hb.json` for Hb. Sampling accepts EEG `.npy` arrays or an NPZ containing `eeg`; it does not require observed target arrays. Use a new output path.

The inherited `forward_with_cfg` method is retained for source equivalence but is unsupported. The training and sampling workflow calls ordinary conditional `forward`.

## Target preparation

### fMRI surface coordinates

The original loader removes rows 2274 and 2275 from the 2,562-row encoded EEG. It applies `(frame / 5 + 1) / 2` to each target frame before removing the same rows.

The sampling CLI returns 2,560 processed coordinates. [data.py](../src/unifiedneurogen/data.py) provides the explicit conversions:

| Helper | Operation |
| --- | --- |
| `remove_legacy_vertices(eeg, target=None)` | Remove the two excluded rows and, when supplied, scale and reshape the target |
| `restore_legacy_vertices(generated, physical_scale=False)` | Insert the excluded nodes as zeros; set `physical_scale=True` to apply the inverse target scaling before insertion |

The original sampler restores the excluded nodes as zeros. A generated vector remains in its declared representation until an explicit transform is applied; conversion to another unit or coordinate system requires its own documented transform.

### Hb tokens

The original Hb target consists of stacked, normalized HbO/HbR tokens. Its `[144, 128]` coordinates differ from M06's 144 mean/slope response coordinates. Raw EEG and M06's 1,575-feature inputs are also different interfaces.

<details>
<summary><strong>Original Hb loader and stored-output layout</strong></summary>

1. `gdit/dataloader.py` loads same-named EEG, HbO and HbR arrays.
2. It stacks HbO/HbR along axis 1, then uses one sample-wise minimum and maximum across the stacked target.
3. `train_with_labram.py` permutes a batch with `(0, 2, 1, 3)` and reshapes it to `[batch, 144, 128]`. It reshapes the EEG condition to `[batch, 144, 1000]`.
4. The original `sample_lm.py` reshapes each generated `[144, 128]` array to `[2, 36, 256]` for storage.

The final reshape does not invert concentration calibration. The source does not save the per-sample extrema needed to invert min/max normalization. It also has no constant-target guard, so prepared targets must be finite after normalization.

</details>

## Checkpoints and software configurations

| Resource | Availability and use |
| --- | --- |
| Upstream original DiT demo checkpoint | Belongs to the fMRI diffusion architecture; not bundled in the small Release demo |
| Pretrained Hb DiT checkpoint | Not located or bundled; training requires compatible original encoded conditions and target arrays |
| [Small Release demo](https://github.com/xkoo115/UnifiedNeuroGen/releases/tag/v0.2.0) | Uses the separate M10 conditional Gaussian model |
| `DEMO_CONFIG` in each diffusion module | Small random models for software validation; no trained neuroimaging generation |
| [dit_smoke.json](../configs/dit_smoke.json) | Reduced fMRI configuration for checking the DDPM/DDIM software path |

Upstream encoders and missing original preprocessing inputs are not reconstructed by this release. Extraction hashes and AST comparisons are recorded in [source_provenance.json](source_provenance.json). Copyright and upstream licence notices are retained; see [sources and licences](sources.md).

---

[Methods](methods.md) · [Data formats](data.md) · [Environment](environment.md) · [Sources and licences](sources.md)
