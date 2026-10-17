# Prepared input formats

All NPZ arrays must load with `allow_pickle=False`. Fitting examples use a one-dimensional Unicode `split` array containing `fit`, `tune` and, optionally, `evaluate`. Participant roles must be established before fitting; related frames and runs must retain the original participant boundaries.

## M10 BOLD generator

| Array | Shape | Meaning |
| --- | --- | --- |
| `eeg_features` | `[N, 2928]` | Corrected EEG log-power, ordered as eight time tokens of 366 features (61 channels by six spectral groups) |
| `targets` | `[N, 5124]` | Run-centred prepared surface BOLD, required for fitting only |
| `split` | `[N]` | Original fitting, tuning or evaluation membership |

The saved fold transform contains EEG feature mean/SD, vertex mean/SD, PCA mean/components and latent mean/SD. These parameters were estimated using the original fitting pool. EEG is standardized without clipping. Target standardization and inverse projection are implemented in `data.py` with the original operation order.

`m10-sample` accepts a bare EEG `.npy` or an `.npz` containing `eeg_features`; it never requests observed targets, target states, task labels or participant IDs. A checkpoint carries its fitted linear anchor as buffers. Inference emits `mean`, `marginal_variance`, `activity`, `hrf_weights`, `surface_mean`, and optionally `samples` with shape `[draws, N, 64]`.

`m10-train` expects the supplied transform to have been fitted using fitting participants only. It refits its Ridge anchor and four auxiliary KMeans states on the selected fitting rows. Tuning reconstruction error selects the saved model state, including epoch zero. It does not select from evaluation scores. Fitting a new transform or constructing physical EEG features from raw recordings is upstream of this core training interface.

The separate real demo contains 48 fixed original rows and the full original fold transformation. Its small fitting subset does not represent a complete new experimental cohort.

## Original diffusion generators

Training NPZ files contain `eeg`, `target` and `split`. Sampling accepts an EEG `.npy` or an NPZ containing `eeg`.

| Branch | `eeg` | `target` |
| --- | --- | --- |
| Original fMRI | `[N, 2560, 1200]` | `[N, 1, 2560]` |
| Original Hb | `[N, 144, 1000]` | `[N, 144, 128]` |
| Small fMRI software configuration | `[N, 2560, 32]` | `[N, 1, 32]` |

The original BOLD loader drops surface indices 2274 and 2275 from 2,562-point encodings/targets and maps the target using `(BOLD / 5 + 1) / 2`. Helpers `remove_legacy_vertices` and `restore_legacy_vertices` preserve this conversion. The sampling CLI returns the model's 2,560 coordinates; restoration and inverse scaling are explicit choices in these helpers.

The original Hb loader stacks preprocessed HbO and HbR tokens, applies per-sample stack-wide min/max normalization and reshapes targets into `[144, 128]`. Inputs are the precomputed EEG encodings expected by the original adaptor. These tokens differ from the 144 physical mean/slope coordinates used by M06. The optional upstream encoders, their training data and missing original preprocessing inputs are not reconstructed by this release.

## M06 Hb response generator

Fitting files contain `eeg_features` `[N,1575]`, `targets` `[N,144]`, `tasks` `[N]` and `split`. Sampling files need only `eeg_features` and `tasks`.

`tasks` contains `MI` or `MA`, identifying the acquisition paradigm. It does not contain left/right imagery, arithmetic/rest or other within-paradigm class labels. EEG features summarize the 0-10-second response. Targets concatenate HbO/HbR means and slopes over 5-15 seconds, with the original -3 to 0-second baseline. The model outputs these response summaries, not a full haemoglobin waveform.

The fixed configuration in `configs/hb.json` is an example within the original candidate family; manuscript configurations were selected separately per fold. The core `fit` API leaves candidate selection to the caller. Load joblib models only from a trusted source.

## M01 and M02 regional generators

`linear.predict_m01(history, dynamic_head, scale_head)` accepts one complete retained run of physical EEG histories `[T,8,366]` and the original `new_dynamic_head.joblib` / `new_functional_head.joblib` objects. It generates 142 ROI time courses, using EEG-predicted log-SD for amplitude and the generated sequence's own mean/SD for normalization. Whole-run EEG context makes this an offline interface.

`linear.predict_m02(history, model, regional_relative=True)` accepts the original `fold*_models_and_scalers.joblib` object and returns fitting-standardized regional predictions, with the frame's regional mean removed when requested. `weighted_ridge_fit` provides the shared equal-person-weight fitting primitive. Downstream selection/readout drivers and these models' checkpoints are not included in the small demo.
