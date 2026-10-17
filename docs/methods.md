# Method and result correspondence

This core release follows the 15 October 2026 manuscript. Its formulation relates EEG-conditioned activity representations to target-specific haemodynamic coordinates; the experimental implementations retain their individual identities.

## Revised BOLD model: M10

M10 uses a fitted linear anchor plus a cross-attention residual to parameterize a diagonal-plus-low-rank Gaussian over 64 fitting-PCA target coordinates. Temporal banks are learned `softmax(hrf_logits)` weights over the EEG history tokens. A fixed Gaussian temporal kernel is not part of this model.

The objective is Gaussian negative log likelihood plus 0.10 symmetric paired activity alignment, 0.20 EEG-state cross entropy, 0.20 target-state cross entropy and 0.01 squared inner product between normalized activity and context representations. The four auxiliary states come from KMeans fitted to the first eight target PCA coordinates in the fitting pool. Observed targets and these state labels are used in fitting; EEG alone supplies the generated distribution at inference.

The included historical configuration has width 128, four attention blocks, eight heads, 64 activity and 64 context coordinates, four covariance factors, dropout 0.1 and an activity/context bottleneck for the mean. Training uses AdamW with learning rate 3e-4, weight decay 2e-4 and gradient norm clipping at 1. The portable runner preserves the core objective and tuning-MSE checkpoint rule. A reduced refit is distinct from retraining all original folds with their complete upstream transformations and hyperparameter choices.

## Hb response model: M06

M06 fits a pooled EEG scaler/PCA and PLS encoder to both MI and MA, using separate 144-coordinate target supervision blocks. Each acquisition paradigm has its own regularized response head. Those two heads share the pooled EEG projection within this branch. The BOLD and Hb cohorts do not establish a single jointly fitted three-modality weight set.

## Original diffusion models

The original BOLD and Hb DiT denoisers condition on precomputed EEG encodings. Their noise-prediction / learned-variance objectives and DDIM transitions are retained from the original implementation. The two modalities have different adaptor and target shapes. Parameters and the original conditioning operations are preserved, including the historical cross-attention implementation. Classifier-free guidance is not exposed by the portable CLI.

## Regional generation: M01 and M02

M01 combines a weighted Ridge dynamic head with a separate EEG-to-run-log-SD head; generated dynamics are centred and scaled within the retained run. M02 is a separately fitted regional linear mapping. `linear.py` includes their generation operations and a fixed-penalty fitting primitive. A fixed double-gamma candidate in these linear models is distinct from the learned temporal banks in M10.

## Figure boundaries

| Result family in the 1015 revision | Implementation/source identity |
| --- | --- |
| Revised regional dynamics/scale results in Fig. 2 | Mainly M01, with a separate M02 regional-relative result |
| Retained waveform, cortical/network and external-localization illustrations | Original archived results; their full checkpoint-to-panel provenance is not established by the included M10 demo |
| Physical Hb response preservation in Fig. 4 | M06 response features; distinct from retained Hb waveform illustrations |
| BOLD representation/readout and augmentation results | M10-based sources plus experiment-specific transformations and downstream readers |
| Retrieval, calibrated intervals and directional transfer | Separate fitted calibration/analysis models |

This package provides generation code and a real historical checkpoint example. It does not claim that one checkpoint reproduces every panel or that withheld downstream experiment drivers are present. The source manifests record the retained core's file identity and extraction checks.
