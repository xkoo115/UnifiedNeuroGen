# Third-party notices

The original UnifiedNeuroGen MIT licence (2025 xkoo115) is retained in the root `LICENSE`. It does not replace the licences of incorporated upstream material.

| Material | Attribution and source | Terms |
| --- | --- | --- |
| Diffusion Transformer architecture and adapted model portions | Meta Platforms, Inc. and affiliates; [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | [CC BY-NC 4.0](licenses/DiT-CC-BY-NC-4.0.txt) |
| Gaussian diffusion, timestep respacing and likelihood utilities | OpenAI; [guided-diffusion](https://github.com/openai/guided-diffusion), [improved-diffusion](https://github.com/openai/improved-diffusion), [GLIDE](https://github.com/openai/glide-text2im) | MIT notices in `licenses/` |
| Attention/MLP code retained through the original EEG model source | [LaBraM](https://github.com/935963004/LaBraM) | [MIT](licenses/LaBraM-MIT.txt) |
| Optional separately distributed data derivatives | NATVIEW data contributors | CC BY 4.0, with attribution in the demo package |

The diffusion modules and model dependency closures were extracted, cleaned and reformatted for this release. Their retained scientific operations were checked against the local original source. Unused source encoders and inactive experiment code were removed. Portable factories, array interfaces and training/sampling commands were added. Copyright and upstream attribution notices remain in the relevant source files.
