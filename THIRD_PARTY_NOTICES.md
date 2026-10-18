[← Repository](README.md) · [Documentation](docs/README.md)

# Licences and attribution

The original UnifiedNeuroGen [MIT licence](LICENSE) (2025 xkoo115) is retained. Incorporated upstream material keeps its own terms.

## Code

| Material | Attribution and source | Terms |
| --- | --- | --- |
| Diffusion Transformer architecture and adapted model portions | Meta Platforms, Inc. and affiliates; [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | [CC BY-NC 4.0](licenses/DiT-CC-BY-NC-4.0.txt) |
| Gaussian diffusion, timestep respacing and likelihood utilities | OpenAI; [guided-diffusion](https://github.com/openai/guided-diffusion), [improved-diffusion](https://github.com/openai/improved-diffusion), [GLIDE](https://github.com/openai/glide-text2im) | MIT notices in `licenses/` |
| Attention/MLP code retained through the original EEG model source | [LaBraM](https://github.com/935963004/LaBraM) | [MIT](licenses/LaBraM-MIT.txt) |

The diffusion modules retain the original model operations and parameter names. Configuration factories, prepared-array interfaces and training/sampling commands were added for this package. Copyright and attribution notices remain in the source files.

## Demo data

The separately distributed demo includes derivatives of NATVIEW data, with attribution to the NATVIEW contributors under **CC BY 4.0**. Data attribution and preparation details are supplied in the demo's `README.md` and `DATA_LICENSE.md`.

Download the package from the [v0.2.0 Release](https://github.com/xkoo115/UnifiedNeuroGen/releases/tag/v0.2.0).

## Manuscript figure

The README displays the [original manuscript Figure 1](assets/paper-framework.pdf). Its anatomical head surface is adapted from **Servier Medical Art by Servier**, under **CC BY 3.0**. This attribution is retained from the manuscript caption.

## Source records

See [sources and verification](docs/sources.md) for implementation origins and the corresponding file-hash records.
