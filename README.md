# UnifiedNeuroGen

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.02433-b31b1b.svg)](https://arxiv.org/abs/2506.02433)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper: **"Empowering Functional Neuroimaging: A Pre-trained Generative Framework for Unified Representation of Neural Signals."**

> **Abstract:** *Multimodal functional neuroimaging enables systematic analysis of brain mechanisms and provides discriminative representations for brain-computer interface (BCI) decoding. However, its acquisition is constrained by high costs and feasibility limitations. Moreover, underrepresentation of specific groups undermines fairness of BCI decoding model. To address these challenges, we propose a unified representation framework for multimodal functional neuroimaging via generative artificial intelligence (AI). By mapping multimodal functional neuroimaging into a unified representation space, the proposed framework is capable of generating data for acquisition-constrained modalities and underrepresented groups. Experiments show that the framework can generate data consistent with real brain activity patterns, provide insights into brain mechanisms, and improve performance on downstream tasks. More importantly, it can enhance model fairness by augmenting data for underrepresented groups.*

## 📝 About The Project

**UnifiedNeuroGen** addresses the significant challenges in neuroimaging: the high cost and limited accessibility of advanced modalities like fMRI, and the resulting fairness issues in AI models due to biased or underrepresented data.

This project introduces a generative AI framework built on these core ideas:
- 🧠 **Unified Representation**: It learns to map diverse neural signals (e.g., low-cost EEG and high-cost fMRI) into a shared, unified feature space.
- 🧬 **Cross-Modal Generation**: It uses a pre-trained **Diffusion Transformer (DiT)** model to generate high-fidelity, high-cost neuroimaging data (like fMRI or fNIRS) from low-cost, easily accessible signals (like EEG).
- ✨ **Enhanced Fairness and Accessibility**: By synthesizing data for underrepresented groups or tasks, the framework improves the fairness of downstream BCI decoding models and dramatically lowers the barrier to entry for advanced neuroimaging research.

![Framework Diagram](./assets/fig1.png)
> A schematic of the framework, illustrating how a low-cost modality (EEG) is transformed into high-cost neuroimaging signals through the unified representation learning framework.

---

If you require detailed implementation, please contact xuhangc@hzu.edu.cn for complete code.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Our model architecture is inspired by the work of the [DiT](https://github.com/facebookresearch/DiT) authors.
- We thank the contributors of the public datasets used in this study.
