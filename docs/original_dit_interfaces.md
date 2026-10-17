# Original diffusion interfaces

These two architectures preserve the author implementations. Their target and condition coordinates differ. Their factories validate the declared condition width because the original `Adaptor` preserves its last feature dimension.

| Branch | Target / initial noise | EEG condition | Denoiser output | Production backbone |
|---|---|---|---|---|
| fMRI DiT | `[batch, 1, 2560]` | `[batch, 2560, 1200]` | `[batch, 2, 2560]` | 24 blocks, width 1200, 12 heads, patch 4 |
| Hb DiT | `[batch, 144, 128]` | `[batch, 144, 1000]` | `[batch, 288, 128]` | 28 blocks, width 1000, 10 heads, patch 1 |

The output channel halves predict noise and learned diffusion variance. Both branches use the original noise MSE plus variational-bound loss, a 1000-step linear DDPM schedule, and conditional DDIM sampling with `eta=0`. Diffusion timesteps are separate from physiological time. The original models do not contain M10's learned history banks or the linear models' fixed double-gamma history candidate.

For the original fMRI loader, encoded EEG has 2562 rows before rows 2274 and 2275 are removed. The target frame receives `(frame / 5 + 1) / 2` before those same rows are removed. The original sampler restores the two excluded nodes as zeros. A sampled vector remains in the declared processed representation; conversion to another unit or coordinate system requires its own documented transform.

For Hb, `gdit/dataloader.py` loads same-named EEG, HbO and HbR arrays. It stacks HbO/HbR along axis 1, then applies a single sample-wise minimum and maximum over the stacked target. `train_with_labram.py` permutes a batch with `(0, 2, 1, 3)` and reshapes it to `[batch, 144, 128]`; it reshapes the condition to `[batch, 144, 1000]`. The original `sample_lm.py` reshapes each generated `[144, 128]` array to `[2, 36, 256]` for storage. This operation is a reshape, not an inverse concentration calibration. The source does not save the per-sample extrema for an inverse min–max transform. The loader also has no guard for a constant target; prepared targets must therefore be finite after normalization.

The upstream original DiT demo checkpoint belongs to the fMRI diffusion architecture. The small demo distributed alongside this release uses M10. A pretrained Hb DiT checkpoint has not been located or bundled. Hb training requires the original compatible encoded condition and target arrays. Raw EEG and M06's 1575-feature/144-response vectors are different interfaces.

`DEMO_CONFIG` in each module creates a small random model for software validation. It does not supply trained neuroimaging generation. The inherited `forward_with_cfg` method is retained for source equivalence; the supported train/sample workflow calls ordinary conditional `forward`.

Extraction hashes and AST comparisons are recorded in [source_provenance.json](source_provenance.json). Copyright and upstream license notices are retained in the release.
