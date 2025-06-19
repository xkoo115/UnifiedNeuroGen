# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 [Your Name or Your Organization's Name]. All rights reserved.
#
# This source code is licensed under the [Specify Your License, e.g., MIT] license found in the
# LICENSE file in the root directory of this source tree.
#
# Author(s): xkoo115
#

"""
UnifiedNeuroGen Project - Inference Script

This script provides the sampling/inference code for the research paper:
"Empowering Functional Neuroimaging: A Pre-trained Generative Framework for Unified Representation of Neural Signals"

It is designed to generate neuroimaging data representations conditioned on EEG signals
using a pre-trained Diffusion Transformer (DiT) model.

The core functionality involves:
1. Loading a pre-trained DiT model checkpoint.
2. Preparing a dataset of EEG signal encodings.
3. Using a DDIM sampler from the diffusion model to generate new data samples
   based on the conditional EEG input.
4. Saving the generated representations for further analysis and evaluation.

To run this script, provide the necessary command-line arguments for the model,
checkpoint path, EEG data path, and the desired output directory.

Example usage:
python sample_lm.py --model DiT_fMRI --ckpt /path/to/your/checkpoint.pt --eeg-path /path/to/eeg/encodings --save-path /path/to/save/generated/data

If you use this code in your research, please consider citing our paper.
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from models import DiT_models
from glob import glob
import numpy as np
import argparse
import cv2
import os
from torch.utils.data import DataLoader, Dataset
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def save_PETimage(tensor, file_name):
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.detach().cpu().numpy()
    tensor = tensor * 255
    cv2.imwrite(file_name, tensor)

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint

class Test_Loader_Nat(Dataset):
    def __init__(self, eeg_encoding_path, generated_path): # The naming method of generated data is as similar as possible to the eeg_encoding
        eeg_paths = [i.split("/")[-1] for i in glob(eeg_encoding_path)]
        already = [i.split("/")[-1] for i in glob(generated_path)]
        common = list(set(eeg_paths).difference(already))
        self.eeg_paths = [f"{eeg_encoding_path}/{i}" for i in common]
        self.remove_indices = [2274, 2275]

    def __len__(self):
        return len(self.eeg_paths)

    def __getitem__(self, item):
        eeg_path = self.eeg_paths[item]
        eeg_data = np.load(eeg_path)
        eeg_data = np.delete(eeg_data, self.remove_indices, axis=0)
        name = eeg_path.split('/')[-1]
        return {'eeg':eeg_data, 'name':name}


def main(args):

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = 2560
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes, in_channels = 1).to(device)
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    b_s = 64
    test_dataset = Test_Loader_Nat(args.eeg_path, args.save_path)
    test_loader = DataLoader(test_dataset, batch_size=b_s, shuffle=False)
    print(f"sample totaly {len(test_dataset)} imgs")
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            cond = sample['eeg'].to(device).float()
            names = sample['name']

            z = torch.randn(cond.shape[0], 1, latent_size, device=device)
            model_kwargs = {'c': cond}
            samples = diffusion.ddim_sample_loop(model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, eta=0.0)

            for name, sample in zip(names, samples):
                sample = sample.cpu().detach().numpy().squeeze()
                sample = np.insert(sample, 2274, 0, axis=0)
                sample = np.insert(sample, 2275, 0, axis=0)

                np.save(f'{args.save_path}/{name}', sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT_fMRI")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a model checkpoint.")
    parser.add_argument("--eeg-path", type=str, default=None, help="Optional path to eeg encoding.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to generated data.")
    args = parser.parse_args()
    main(args)
