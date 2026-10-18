[← Repository](../README.md) · [Documentation](README.md)

# Prepared data formats

Array names, shapes and fitted transformations for each generation interface. For the model-to-manuscript mapping, see [methods and model correspondence](methods.md).

[M10 BOLD](#m10-bold-generator) · [Original DiT](#original-diffusion-generators) · [M06 Hb](#m06-hb-response-generator) · [M01 / M02](#m01-and-m02-regional-generators)

## Common conventions

All NPZ arrays must load with `allow_pickle=False`. Fitting files use a one-dimensional Unicode `split` array containing `fit`, `tune` and, optionally, `evaluate`. Establish participant roles before fitting; related frames and runs must retain their original participant boundaries.

`N` denotes examples and `T` denotes frames in one retained run. Each branch uses its own prepared features, target coordinates and fitted model objects. Files prepared for one branch are not interchangeable with another branch's inputs.

## M10 BOLD generator

### Training inputs

| Array | Shape | Contents |
| --- | --- | --- |
| `eeg_features` | `[N, 2928]` | Corrected EEG log-power: eight time tokens of 366 features, each comprising 61 channels × six spectral groups |
| `targets` | `[N, 5124]` | Run-centred prepared surface BOLD |
| `split` | `[N]` | Original fitting, tuning or evaluation membership |

The saved fold transform contains EEG feature mean/SD, vertex mean/SD, PCA mean/components and latent mean/SD, estimated from the original fitting pool. EEG is standardized without clipping. Target standardization and inverse projection follow the original operation order in [data.py](../src/unifiedneurogen/data.py).

`m10-train` requires a supplied transform fitted using fitting participants only. It refits the Ridge anchor and four auxiliary KMeans states on the selected fitting rows. Tuning reconstruction error selects the saved state, including epoch zero; evaluation scores do not select the model. Fitting a new transform and constructing physical EEG features from raw recordings are upstream of this core interface. See [M10 training settings](methods.md#conditional-bold-generation-m10) for the defaults.

### Inference inputs and outputs

`m10-sample` accepts an EEG `.npy` file or an `.npz` containing `eeg_features`, together with the fitted transform and checkpoint. Observed targets, target states, task labels and participant IDs are not required. The checkpoint carries its fitted linear anchor as buffers.

| Output field | Contents |
| --- | --- |
| `mean` | Conditional mean in 64 standardized target PCA coordinates |
| `marginal_variance` | Marginal variance in those coordinates |
| `activity` | EEG-derived activity representation |
| `hrf_weights` | Learned temporal weights |
| `surface_mean` | Conditional mean projected to the 5,124-value prepared surface |
| `samples` | Optional Gaussian draws, shaped `[draws, N, 64]` |

The separate [demo Release](https://github.com/xkoo115/UnifiedNeuroGen/releases/tag/v0.2.0) contains 48 fixed original rows and the full original fold transformation. Its small fitting subset does not represent a complete new experimental cohort. The outputs remain in run-centred prepared coordinates and do not restore removed run means.

## Original diffusion generators

### Training and inference inputs

Training NPZ files contain `eeg`, `target` and `split`. Sampling accepts an EEG `.npy` file or an NPZ containing `eeg`.

| Configuration | `eeg` | `target` |
| --- | --- | --- |
| Original fMRI | `[N, 2560, 1200]` | `[N, 1, 2560]` |
| Original Hb | `[N, 144, 1000]` | `[N, 144, 128]` |
| Small fMRI software configuration | `[N, 2560, 32]` | `[N, 1, 32]` |

These inputs are precomputed EEG encodings for the original adaptors. Optional upstream encoders, their training data and missing original preprocessing inputs are not reconstructed by this release. The small software configuration uses a random model; the downloadable trained demo uses **M10**.

### Target preparation and output coordinates

| Branch | Preparation | Sampling output |
| --- | --- | --- |
| fMRI | Remove surface indices 2274 and 2275 from 2,562-point encodings and targets; map the target using `(BOLD / 5 + 1) / 2` | The CLI returns the model's 2,560 coordinates. Restoration and inverse scaling are explicit helper operations. |
| Hb | Stack preprocessed HbO/HbR tokens, apply per-sample stack-wide min/max normalization, then reshape to `[144, 128]` | Stacked haemoglobin tokens in the processed representation, distinct from M06's 144 physical mean/slope coordinates |

For fMRI, `remove_legacy_vertices` and `restore_legacy_vertices` in [data.py](../src/unifiedneurogen/data.py) implement the coordinate conversion. For the original Hb loader order, output reshaping and checkpoint requirements, see [original DiT interfaces](original_dit_interfaces.md).

## M06 Hb response generator

### Training and inference inputs

| Array | Shape | Required for | Contents |
| --- | --- | --- | --- |
| `eeg_features` | `[N, 1575]` | Training and inference | EEG response features summarizing 0–10 seconds |
| `targets` | `[N, 144]` | Training | Concatenated HbO/HbR means and slopes over 5–15 seconds, using the original −3 to 0-second baseline |
| `tasks` | `[N]` | Training and inference | Acquisition paradigm: `MI` or `MA` |
| `split` | `[N]` | Training | Original participant-level sample roles |

`tasks` identifies motor imagery or mental arithmetic. It does not contain left/right imagery, arithmetic/rest or other within-paradigm class labels. Fitting requires both acquisition paradigms. The output contains response summaries rather than a full haemoglobin waveform.

### Fitted model

The model pools the EEG scaler/PCA/PLS projection across MI and MA and fits a separate response head for each paradigm. This shared projection belongs to M06; the BOLD models retain their own transformations and weights.

[configs/hb.json](../configs/hb.json) supplies a fixed example within the original candidate family. Manuscript configurations were selected separately per fold; the core `fit` API leaves candidate selection to the caller. Load joblib models only from a trusted source.

## M01 and M02 regional generators

These Python interfaces consume physical EEG histories with shape `[T, 8, 366]` and the original saved-head objects.

| Interface | Saved objects | Output and context |
| --- | --- | --- |
| `linear.predict_m01(history, dynamic_head, scale_head)` | `new_dynamic_head.joblib` and `new_functional_head.joblib` | 142 ROI time courses for one complete retained run. EEG-predicted log-SD sets amplitude; the generated sequence's own mean/SD sets normalization. Whole-run EEG context makes this an offline interface. |
| `linear.predict_m02(history, model, regional_relative=False)` | `fold*_models_and_scalers.joblib` | Fitting-standardized regional predictions. Set `regional_relative=True` to remove each frame's regional mean. |

`weighted_ridge_fit` provides the shared equal-person-weight fitting primitive. Downstream selection/readout drivers and the M01/M02 checkpoints are not included in the small demo. See [regional methods](methods.md#regional-bold-generation-m01-and-m02) for their relationship to the manuscript.

---

[Methods](methods.md) · [Original DiT interfaces](original_dit_interfaces.md) · [Environment](environment.md) · [Sources and licences](sources.md)
