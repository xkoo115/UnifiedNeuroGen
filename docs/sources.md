[← Repository](../README.md) · [Documentation](README.md)

# Sources and verification

Implementation sources, recorded compatibility checks and file-integrity manifests.

[Source records](#source-records) · [Recorded comparisons](#recorded-comparisons) · [File integrity](#file-integrity)

## Source records

| Implementation | Research source | Detailed record |
| --- | --- | --- |
| Original BOLD DiT and diffusion utilities | Public UnifiedNeuroGen revision [`427a4f3`](https://github.com/xkoo115/UnifiedNeuroGen/tree/427a4f342ebd0494690a6b4c31186c2084b9b22a) | [Original source record](source_provenance.json) |
| Original Hb DiT | Author-supplied `gdit/models.py` archive; identified by its file hash, without a separately verified public Git revision | [Hb archive record](source_provenance.json) |
| M10 | `src/neurovascular_decoder/latent_decoder.py` and the associated BOLD experiment runner | [M10 / M06 record](source_provenance_m10_hb.json) |
| M06 | `scripts/practice_shared_hb_functional_20261001.py` | [M10 / M06 record](source_provenance_m10_hb.json) |
| M01 / M02 | Original regional-generation runners, saved fitted heads and prediction arrays | [Regional model record](source_provenance_linear.json) |

The JSON files contain source paths, source hashes and hashes for nine published implementation files. Paths to research runners identify their original files; those runners and their experiment data are separate from this generation package. Model-to-equation and model-to-figure assignments are in [methods](methods.md).

## Recorded comparisons

These comparisons check the implementations against their research sources. Their numerical scope is recorded in the linked JSON files.

| Check | Recorded result | Scope |
| --- | --- | --- |
| Original DiT operations | Retained Python syntax trees agree after excluding docstrings | Original model nodes and four diffusion utility files |
| Original DiT forward pass | Maximum absolute difference `0` | Matched small fMRI and Hb configurations, identical initialization and strict state loading |
| M10 source replay | Forward outputs, sampled outputs, loss and parameter gradients agree | Historical decoder and its fitted checkpoint |
| M10 demo reference | Maximum absolute mean difference `2.384185791015625e-7` | 48 supplied EEG examples |
| M06 fixed-configuration fit and prediction | Maximum absolute difference `0` | Recorded response data and selected fitting configuration |
| M01 saved predictions | Maximum absolute difference `2.525039614020841e-5`, within `3e-5` tolerance | One fixed evaluation run, all 103 retained frames |
| M02 saved predictions | Maximum absolute difference `7.435781934628238e-9` | The same recorded evaluation run |
| M02 regional-relative predictions | Maximum absolute difference `1.3533513490426863e-8`, within `1e-7` tolerance | The same recorded evaluation run |

The M10 architecture supports low-rank covariance factors. The supplied checkpoint has zero factors and therefore a diagonal conditional covariance. M06 predicts response summaries; its source comparisons do not establish full-waveform reconstruction.

Complete checkpoint-to-panel replay has not been established for every retained manuscript illustration. See the [result-group mapping](methods.md#models-used-by-manuscript-result-groups) for what is provided by each implementation.

## File integrity

From the repository root, verify the code and documentation files against [`SHA256SUMS`](../SHA256SUMS):

```bash
shasum -a 256 -c SHA256SUMS
```

On Linux, `sha256sum -c SHA256SUMS` is an alternative. The manifest covers every tracked file except itself. The separate demo ZIP has its own download checksum, and its extracted files have a separate manifest checked by the `demo` command.

## Software checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The [GitHub Actions workflow](https://github.com/xkoo115/UnifiedNeuroGen/actions/workflows/tests.yml) reports checks for current commits. The source JSON files preserve the comparisons recorded when the implementations were packaged; their test counts are historical records.

For data shapes and coordinate conventions, see [data formats](data.md). For upstream terms, see [licences and attribution](../THIRD_PARTY_NOTICES.md).
