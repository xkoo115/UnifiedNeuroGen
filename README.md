# UnifiedNeuroGen

Core code for **A generative framework for unified representations bridging electrophysiological and haemodynamic brain activity**.

EEG-conditioned generation of BOLD and haemoglobin representations, with the original diffusion implementation and the target-specific generators used in the revised work. This release concentrates on generation: model fitting, saved transformations and inference. The demo data and checkpoint are distributed separately.

## Installation

Python 3.10 or newer is required. The small demo runs on a CPU; the original full DiT models require substantially more memory and are intended for GPU use.

```bash
git clone https://github.com/xkoo115/UnifiedNeuroGen.git UnifiedNeuroGen
cd UnifiedNeuroGen
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The versions used for local validation are recorded in [docs/validated_environment.txt](docs/validated_environment.txt).

## Quick start

Download the real demo data and checkpoint from the [v0.2.0 Release](https://github.com/xkoo115/UnifiedNeuroGen/releases/tag/v0.2.0). The demo ZIP is about 8.7 MB and is distributed as a Release attachment, separately from Git history.

From the repository root, download and verify the ZIP before extracting it:

```bash
curl -fL --retry 3 -o UnifiedNeuroGen-demo.zip \
  https://github.com/xkoo115/UnifiedNeuroGen/releases/download/v0.2.0/UnifiedNeuroGen-demo.zip
curl -fL --retry 3 -o UnifiedNeuroGen-demo.zip.sha256 \
  https://github.com/xkoo115/UnifiedNeuroGen/releases/download/v0.2.0/UnifiedNeuroGen-demo.zip.sha256
shasum -a 256 -c UnifiedNeuroGen-demo.zip.sha256
unzip UnifiedNeuroGen-demo.zip -d ..
```

On Linux, `sha256sum -c UnifiedNeuroGen-demo.zip.sha256` is an alternative checksum command. The extracted layout is:

```text
UnifiedNeuroGen/
UnifiedNeuroGen-demo/
```

From the repository root, run:

```bash
python -m unifiedneurogen demo \
  --data-dir ../UnifiedNeuroGen-demo \
  --output outputs/demo
```

The command checks the demo's SHA-256 manifest, loads the historical M10 fold-01 seed-7 checkpoint, and generates 64-dimensional conditional means and Gaussian samples from 48 real EEG feature vectors. It also projects the means to the 5,124-value prepared surface coordinates. The reference BOLD arrays are unnecessary for inference. See `outputs/demo/report.json` and `outputs/demo/generated.npz`.

A successful run reports `"status": "PASS"`, `"samples": 48`, `"latent_shape": [48, 64]`, `"surface_shape": [48, 5124]` and `"target_required_at_inference": false`. The CPU computation takes approximately one second on the validated machine, excluding installation and startup. To repeat it, choose a fresh output directory, such as `--output outputs/demo_second_run`.

The surface values are in the run-centred, preprocessed target coordinates. They do not restore the removed run means. The small sample illustrates the generation interface; it does not estimate the manuscript's aggregate metrics.

## Implementations

| Module | Input and generated output | Role |
| --- | --- | --- |
| `dit.py` + `diffusion/` | Surface EEG encodings to BOLD surface values through DDIM | Original EEG-to-fMRI diffusion generator |
| `gdit.py` + `diffusion/` | EEG encodings to stacked haemoglobin tokens through DDIM | Original EEG-to-Hb diffusion generator |
| `m10.py` | 2,928 EEG log-power coordinates to a conditional Gaussian over 64 target PCA coefficients | Revised BOLD representation generator; included real demo checkpoint |
| `hb.py` | 1,575 EEG response features to 144 Hb mean/slope coordinates | M06 pooled EEG projection and MI/MA-specific response heads |
| `linear.py` | Prepared EEG histories to regional BOLD dynamics/scale coordinates | M01 and M02 generation primitives and saved-head inference |

The unifying element is the representation workflow. These implementations have distinct target coordinates and fitted weights. In particular, the M10 demo checkpoint is a conditional Gaussian model, and M06 generates response summaries. The original diffusion models use their own encoded inputs.

[Methods and result correspondence](docs/methods.md) describes these identities and their relation to the revised figures. This core release omits downstream decoding, retrieval, significance testing and figure-production drivers.

## Training and generation

The CLI accepts prepared arrays and explicit configuration files; it has no machine-specific dataset paths.

Generate from EEG alone with the supplied M10 checkpoint:

```bash
python -m unifiedneurogen m10-sample \
  --eeg ../UnifiedNeuroGen-demo/samples.npz \
  --transform ../UnifiedNeuroGen-demo/transforms/fold_transform.npz \
  --checkpoint ../UnifiedNeuroGen-demo/weights/m10.pt \
  --config configs/m10.json \
  --output outputs/m10_generated.npz
```

Fit a new M10 model using the example's original sample roles and saved fitting transformations:

```bash
python -m unifiedneurogen m10-train \
  --data ../UnifiedNeuroGen-demo/samples.npz \
  --transform ../UnifiedNeuroGen-demo/transforms/fold_transform.npz \
  --config configs/m10.json \
  --epochs 2 --batch-size 16 \
  --output outputs/m10_training
```

This small refit uses a newly fitted Ridge anchor and four fitting-derived target states. It is a training demonstration, with a user-specified fixed Ridge penalty. The historical checkpoint was fitted on its original full fitting cohort.

The original diffusion branch uses the same portable training and sampling commands:

```bash
python -m unifiedneurogen dit-train \
  --data paired_encodings.npz --config configs/dit_fmri.json \
  --device cuda --output outputs/dit_training

python -m unifiedneurogen dit-sample \
  --eeg eeg_encodings.npy --checkpoint outputs/dit_training/dit.pt \
  --config configs/dit_fmri.json --steps 50 --device cuda \
  --output outputs/dit_generated.npz
```

For the original Hb model, add `--branch hb` and use `configs/dit_hb.json` in both commands. The full original diffusion checkpoint and upstream EEG encoders are not part of the small demo. A reduced random model for checking the DDPM/DDIM software path is configured in `configs/dit_smoke.json`; it carries no scientific performance claim.

Fit and use the M06 response generator with a fixed configuration:

```bash
python -m unifiedneurogen hb-train \
  --data hb_pairs.npz --config configs/hb.json --output outputs/hb_training
python -m unifiedneurogen hb-sample \
  --data hb_eeg.npz --checkpoint outputs/hb_training/hb.joblib \
  --output outputs/hb_generated.npz
```

[Data formats](docs/data.md) specifies array names, shapes and preprocessing boundaries. CLI outputs must be new files or empty directories. For detailed options, run `python -m unifiedneurogen <command> --help`.

## Validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests cover denoising and DDIM sampling, Gaussian likelihoods, learned temporal weights, fitting/inference boundaries and target transformations. Release verification additionally compares the real demo's deterministic means against saved historical predictions. Production-scale retraining and complete figure reproduction are outside these checks.

## Data, citation and licence

The demo includes derivatives of [NATVIEW](https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html), distributed with attribution under CC BY 4.0. Its README records data preparation, original participant roles and checkpoint identity.

Citation metadata is in [CITATION.cff](CITATION.cff). The original project's MIT licence is retained in [LICENSE](LICENSE). Third-party portions retain their own terms, including Meta DiT's CC BY-NC 4.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Earlier implementation

The original implementation and its Git history are preserved in the [legacy backup repository](https://github.com/xkoo115/UnifiedNeuroGen-legacy-backup-20261017) and the [pre-update snapshot](https://github.com/xkoo115/UnifiedNeuroGen/tree/legacy-before-npj-20261017). The repository address cited in the manuscript is retained for this current core release.
