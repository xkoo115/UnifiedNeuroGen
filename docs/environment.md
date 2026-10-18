[← Repository](../README.md) · [Documentation](README.md)

# Environment

Installation requirements, device expectations and the recorded local validation environment.

[Install](#install) · [Devices](#devices) · [Reference versions](#reference-versions)

## Install

Use **Python 3.10 or newer**. From the repository root, create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependencies and version ranges are declared in [`pyproject.toml`](../pyproject.toml). To inspect the available commands:

```bash
python -m unifiedneurogen --help
python -m unifiedneurogen m10-train --help
```

The [quick start](../README.md#quick-start) covers downloading, verifying and running the separate demo.

## Devices

| Workload | Device expectations |
| --- | --- |
| Supplied M10 demo | Runs on a CPU |
| M06 and regional linear models | NumPy / scikit-learn CPU operations |
| Full original BOLD / Hb DiT | Intended for GPU use; substantially larger than the small demo |
| Reduced DiT software configuration | Small random model for testing training and DDIM sampling |

PyTorch commands accept `--device cpu`, `--device cuda` or `--device mps`; the chosen backend must be available in the local PyTorch installation. The recorded validation below uses CPU. The reduced random DiT configuration is a software check, with no trained physiological mapping.

## Reference versions

The following versions were recorded during local validation on **macOS arm64, CPU**. The original plain-text record is [`validated_environment.txt`](validated_environment.txt).

| Component | Recorded version |
| --- | --- |
| Python | `3.12.12` |
| PyTorch | `2.13.0` |
| NumPy | `2.5.3` |
| SciPy | `1.18.1` |
| scikit-learn | `1.9.0` |
| joblib | `1.5.3` |
| tqdm | `4.70.1` |
| threadpoolctl | `3.6.0` |
| pytest | `9.0.2` |

This is an observed reference environment, not a cross-platform lock file. Installation follows the dependency ranges in `pyproject.toml`.

## Run the tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

For the scope of source comparisons and demo checks, see [sources and verification](sources.md).
