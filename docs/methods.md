# Methods and model correspondence

The manuscript uses a common representation workflow: prepare EEG features, fit source and target transformations, generate target-domain coordinates, and apply the corresponding output transformation. The experiments use separate fitted implementations within that workflow. The model names below identify those implementations.

Figure 1 shows the overall framework, including upstream feature extraction. The original diffusion models accept precomputed EEG encodings. M10 uses EEG log-power features and fitted target PCA coordinates; M06 uses EEG response features and a pooled fitted projection. The M10 demo starts from prepared arrays and does not run pretrained EEG or fMRI encoders.

## Conditional BOLD generation: M10

M10 generates a conditional Gaussian over 64 standardized target PCA coordinates from 2,928 EEG log-power features. Its mean combines a fitted Ridge anchor with an attention residual. EEG history contains eight tokens, each with 366 features.

| Manuscript method | Implementation |
| --- | --- |
| Eq. (6): learnable temporal weighting | `MultiResolutionCrossAttentionDecoder.encode_eeg` applies `softmax(hrf_logits)` to the EEG history tokens. These weights are learned rather than fixed Gaussian kernels. |
| Eq. (8): conditional Gaussian | `ConditionedFactorizedDecoder` predicts a mean and diagonal variance, with support for a low-rank covariance factor. |
| Eq. (9): fitted anchor plus attention residual | `ConditionedFactorizedDecoder.forward` adds the fitted linear prediction to the learned residual. |
| Eq. (10): likelihood and representation constraints | `m10_loss` combines Gaussian NLL, symmetric paired activity alignment, EEG and target activity-state cross entropy, and activity/context orthogonality. |

The loss weights are 0.10 for paired alignment, 0.20 for each activity-state cross entropy, and 0.01 for the squared inner product between normalized activity and context codes. The four activity states are fitted by KMeans on the first eight target PCA coordinates from fitting participants. Targets and state labels supervise fitting; inference uses EEG alone.

`configs/m10.json` specifies width 128, four attention blocks, eight heads, 64 activity and context coordinates, dropout 0.1 and four covariance-factor columns. The supplied historical checkpoint has zero covariance factors, so its generated distribution is diagonal Gaussian. The architecture also supports nonzero low-rank factors.

The training command uses the historical settings below. `--seed` controls model initialization and minibatch order; the activity-state seed is fixed separately.

| Setting | Default |
| --- | --- |
| Optimizer | AdamW, learning rate `3e-4`, weight decay `2e-4` |
| Batch size / maximum epochs / patience | `1024` / `12` / `3` |
| Gradient norm clipping | `1.0` |
| Ridge anchor | `alpha=1000`, `solver="lsqr"`, `tol=1e-5` |
| Activity-state clustering | `n_init=20`, `random_state=20260830` |
| Saved model selection | Minimum tuning-participant reconstruction MSE, including epoch zero |

The README's two-epoch, batch-16 example refits the small demo subset. Full-cohort retraining also requires the original fitting-participant transformations and fold data. The supplied fold-01 seed-7 checkpoint retains its original weights and fitted anchor.

## Haemoglobin response generation: M06

M06 implements the pooled response model described with Eq. (12). It fits one EEG scaler/PCA/PLS projection across motor imagery (MI) and mental arithmetic (MA), with a separate regularized response head for each acquisition paradigm. The input `tasks` selects MI or MA; it is not the within-paradigm class label.

Each prediction contains 144 HbO/HbR mean and slope coordinates, using the response windows documented in [data formats](data.md). These are response summaries rather than full haemoglobin waveforms. The shared EEG projection belongs to the M06 branch; the BOLD models have their own fitted transformations and weights. `configs/hb.json` gives one fixed candidate configuration. The original experiments selected configurations per fold.

## Regional BOLD generation: M01 and M02

M01 combines a weighted Ridge dynamic head with an EEG-to-run-log-SD head. Its generated sequence is centred and scaled within the retained run. `linear.predict_m01` therefore takes a complete run of EEG history and provides offline generation.

M02 uses a separately fitted regional linear map. `linear.predict_m02` returns fitting-standardized regional predictions and can remove each frame's regional mean. The fixed double-gamma history candidate in these linear models is separate from M10's learned temporal weighting.

## Original diffusion models

`dit.py` and `gdit.py` retain the original BOLD and Hb DiT operations and checkpoint parameter names. Both use a noise-prediction objective with a learned-variance term in the supplied configurations, followed by conditional DDIM sampling with `eta=0`. The command-line interface also supports fixed variance when `learn_sigma=false` is set in the model configuration.

The two branches have different adaptors, encoded inputs and target shapes. Their historical cross-attention operations are retained for checkpoint compatibility. The inherited `forward_with_cfg` method is unsupported; the training and sampling commands use ordinary conditional `forward`. See [diffusion interfaces](original_dit_interfaces.md) for shapes and preprocessing.

## Models used by manuscript result groups

| Figure or result group | Generator or analysis source | Included here |
| --- | --- | --- |
| Fig. 2c-d, f-h: regional dynamics, scale and generated-signal readout | M01, followed by result-specific aggregation or readout | Generation primitives accepting saved-head objects |
| Fig. 2e: regional-relative correspondence | M02 | Generation primitive accepting saved-head objects |
| Fig. 4c-d: locked Hb decoding and spatial response contrasts | M06, followed by locked readers or response contrasts | Pooled projection and response heads |
| Fig. 5a-b: BOLD decoding | M10 conditional means, followed by separate decoding analyses | M10 generator; downstream readers are separate |
| Fig. 6a-f: low-sample augmentation | Frozen M10 conditional means, followed by augmentation and decoding | M10 generator; augmentation experiments are separate |
| Fig. 4e: bidirectional retrieval | M07 paired feature-space retrieval | Separate analysis |
| Fig. 4f: calibrated intervals | M09 residual calibration | Separate analysis |
| Fig. 4g: directional transfer | M08 baseline CCA and directional readers | Separate analysis |
| Fig. 1, Fig. 2a, Fig. 3 and Fig. 4a-b: retained framework, cortical/network and waveform illustrations | Archived manuscript material | Original framework figure and diffusion source; a complete checkpoint-to-panel replay is not established |
| Fig. 2b: displayed participant traces | Revised trace assembly | Complete trace-to-checkpoint provenance has not been verified |
| Fig. 5c: external localization | Retained NODDI/Parkinson localization analysis | Separate archived analysis |

The demo verifies one M10 checkpoint against saved predictions. Complete figure reproduction also requires the original fold data, model-specific checkpoints and downstream analysis scripts. File hashes and source comparisons are recorded in `source_provenance*.json`.
