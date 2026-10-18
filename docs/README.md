[← Repository](../README.md)

# Documentation

Guides for choosing a generator, preparing inputs and understanding the implementations behind the manuscript results.

## Find the right guide

| I want to… | Start here |
| --- | --- |
| Run the supplied checkpoint on the demo | [Quick start](../README.md#quick-start) |
| Understand the models and their relation to the paper | [Methods and model correspondence](methods.md) |
| Prepare arrays for training or inference | [Data formats](data.md) |
| Work with the original BOLD or Hb diffusion models | [Original diffusion interfaces](original_dit_interfaces.md) |
| Check implementation sources and file integrity | [Sources and verification](sources.md) |
| Set up Python and inspect the reference environment | [Environment](environment.md) |
| Review code, data and figure terms | [Licences and attribution](../THIRD_PARTY_NOTICES.md) |

## Choose an interface

| Model | Prepared input | Generated output | Entry point |
| --- | --- | --- | --- |
| M10 | EEG log-power features | Conditional Gaussian target coordinates and projected BOLD surface means | `demo`, `m10-train`, `m10-sample` |
| M06 | EEG response features and acquisition paradigm | HbO/HbR response means and slopes | `hb-train`, `hb-sample` |
| M01 / M02 | EEG histories and saved fitted-head objects | Regional BOLD coordinates | Python functions in [`linear.py`](../src/unifiedneurogen/linear.py) |
| Original BOLD / Hb DiT | Precomputed EEG encodings | BOLD surface values or Hb tokens | `dit-train`, `dit-sample` |

Each branch has its own input coordinates, fitted transformations and weights. The supplied demo uses **M10**. Model definitions, equations and figure assignments are collected in the [methods guide](methods.md).

## Code and demo layout

The code repository and the extracted demo sit beside one another:

```text
parent/
├── UnifiedNeuroGen/
│   ├── configs/              # Model configurations
│   ├── docs/                 # These guides
│   ├── src/unifiedneurogen/   # Models, transforms and commands
│   └── tests/                # Software tests
└── UnifiedNeuroGen-demo/
    ├── samples.npz
    ├── transforms/
    ├── weights/
    └── SHA256SUMS
```

Download the data and checkpoint from the [v0.2.0 Release](https://github.com/xkoo115/UnifiedNeuroGen/releases/tag/v0.2.0), then follow the [download and run instructions](../README.md#quick-start). The demo checks one historical checkpoint against saved predictions. Complete manuscript analyses also require their original fold data, model-specific checkpoints and downstream scripts.

For software citation, use [CITATION.cff](../CITATION.cff) or the [README citation](../README.md#citation).
