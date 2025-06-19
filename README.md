# UnifiedNeuroGen

[![Paper](https://img.shields.io/badge/Paper-arXiv:2506.02433-b31b1b.svg)](https://arxiv.org/abs/2506.02433)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper: **"Empowering Functional Neuroimaging: A Pre-trained Generative Framework for Unified Representation of Neural Signals."** The paper has now been submitted to nature biomedical engineering.

> **Abstract:** *Multimodal functional neuroimaging enables systematic analysis of brain mechanisms and provides discriminative representations for brain-computer interface (BCI) decoding. However, its acquisition is constrained by high costs and feasibility limitations. Moreover, underrepresentation of specific groups undermines fairness of BCI decoding model. To address these challenges, we propose a unified representation framework for multimodal functional neuroimaging via generative artificial intelligence (AI). By mapping multimodal functional neuroimaging into a unified representation space, the proposed framework is capable of generating data for acquisition-constrained modalities and underrepresented groups. Experiments show that the framework can generate data consistent with real brain activity patterns, provide insights into brain mechanisms, and improve performance on downstream tasks. More importantly, it can enhance model fairness by augmenting data for underrepresented groups.*

## 📝 About The Project

**UnifiedNeuroGen** addresses the significant challenges in neuroimaging: the high cost and limited accessibility of advanced modalities like fMRI, and the resulting fairness issues in AI models due to biased or underrepresented data.

This project introduces a generative AI framework built on these core ideas:
- 🧠 **Unified Representation**: It learns to map diverse neural signals (e.g., low-cost EEG and high-cost fMRI) into a shared, unified feature space.
- 🧬 **Cross-Modal Generation**: It uses a pre-trained **Diffusion Transformer (DiT)** model to generate high-fidelity, high-cost neuroimaging data (like fMRI or fNIRS) from low-cost, easily accessible signals (like EEG).
- ✨ **Enhanced Fairness and Accessibility**: By synthesizing data for underrepresented groups or tasks, the framework improves the fairness of downstream BCI decoding models and dramatically lowers the barrier to entry for advanced neuroimaging research.

![Framework Diagram](./assets/fig1.png)
> **Figure 1**: A schematic of the framework, illustrating how a low-cost modality (EEG) is transformed into high-cost neuroimaging signals through the unified representation learning framework.


## 🚀 Getting Started

Follow these steps to set up and run the project.

### 📦 Prerequisites

This project uses a `requirements.txt` file to manage dependencies. Ensure you have Python and pip installed.
```bash
# Clone the repository
git clone [https://github.com/your-username/UnifiedNeuroGen.git](https://github.com/your-username/UnifiedNeuroGen.git)
cd UnifiedNeuroGen

# Install the required packages
pip install -r requirements.txt
```

### ⚙️ Usage Workflow

#### 1. Prepare Your Data

This project requires paired, preprocessed neural signal data (e.g., EEG and fMRI).
- Open the `dataloader.py` file.
- Locate the `Pair_Loader_Nat` class.
- **Modify the file paths**: Replace the hard-coded placeholder paths, such as `"path to eeg encoding in training set"` and `"path/to/training_set/eeg_encoding/"`, with the actual paths to your preprocessed data.

#### 2. Train the Model

Once your data is ready, run the main training script `train.py` to train the DiT model. The script supports distributed multi-GPU training.

- **For single-GPU training:**
  ```bash
  python train.py --global-batch-size [YOUR_BATCH_SIZE] --results-dir ./results
  ```
- **For multi-GPU distributed training (Recommended):**
  ```bash
  # Example for 2 GPUs
  torchrun --nproc_per_node 2 train.py --global-batch-size [YOUR_BATCH_SIZE] --results-dir ./results
  ```

During training, model checkpoints will be automatically saved in the directory specified by `--results-dir` (defaults to `./results/`).

#### 3. Generate/Sample Data

After the model is trained, use the `sample.py` script to load the checkpoint and generate data for the target modality from input EEG signals.

- Run the following command:
  ```bash
  python sample.py \
    --model DiT_fMRI \
    --ckpt ./results/[EXPERIMENT_FOLDER]/checkpoints/[CHECKPOINT_NAME].pt \
    --eeg-path /path/to/your/test/eeg/encodings \
    --save-path /path/to/save/generated/data
  ```
  - `--ckpt`: Path to your trained model checkpoint file (`.pt`) from Step 2.
  - `--eeg-path`: Path to the test set of EEG encodings you want to use for generation.
  - `--save-path`: Directory where the generated data will be saved.

## 📁 File Structure

- `train.py`: The main script for training the model, with support for distributed training.
- `sample.py`: The script for loading a trained model to perform inference/sampling.
- `dataloader.py`: Defines the data loader. **You must configure your data paths here**.
- `models.py`: Contains the core DiT model architecture.
- `diffusion.py`: Implements the diffusion model's sampling and loss logic.
- `requirements.txt`: A list of all Python dependencies for the project.

## 📜 Citation

If you use this project in your research, please cite our paper:

```bibtex
@misc{yao2025empoweringfunctionalneuroimagingpretrained,
      title={Empowering Functional Neuroimaging: A Pre-trained Generative Framework for Unified Representation of Neural Signals}, 
      author={Weiheng Yao and Xuhang Chen and Shuqiang Wang},
      year={2025},
      eprint={2506.02433},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.02433}, 
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Our model architecture is inspired by the work of the [DiT](https://github.com/facebookresearch/DiT) authors.
- We thank the contributors of the public datasets used in this study.
