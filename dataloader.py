# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 xkoo115. All rights reserved.
#
# This source code is licensed under the [Specify Your License, e.g., MIT] license found in the
# LICENSE file in the root directory of this source tree.
#
# Author(s): xkoo115
#

"""
UnifiedNeuroGen Project - Data Loader and Preprocessing

This script contains the data loading and preprocessing utilities for the UnifiedNeuroGen project.
It defines the PyTorch Dataset class and various normalization functions required for handling
neuroimaging data (EEG and fMRI).

Key Components:
- Preprocessing Functions:
  - eeg_stft: Applies Short-Time Fourier Transform to EEG data.
  - normlize_percent: Normalizes data based on the 98th percentile of positive values.
  - gridnrom: Normalizes data by scaling it to the range of [mean - 3*std, mean + 3*std].
  - z_score_normalize: Applies standard z-score normalization.

- Dataset Class:
  - Pair_Loader_Nat: A PyTorch Dataset designed to load paired EEG and fMRI data samples
    for the model's training process. It handles file path resolution, loads the
    corresponding data pairs, and applies initial transformations.
"""

import numpy as np
from torch.utils.data import Dataset
from glob import glob
import librosa

def eeg_stft(eeg_data, n_fft=256, hop_length=128, win_length=256):
    channels = eeg_data.shape[1]
    stft_results = []
    for i in range(channels):
        # Apply STFT to each channel
        stft_channel = librosa.stft(eeg_data[:, i], n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # Convert to Amplitude
        stft_magnitude = np.abs(stft_channel)
        stft_results.append(stft_magnitude)

    return np.array(stft_results)

def normlize_percent(data):
    positive_values = data[data > 0]

    if positive_values.size > 0:
        pos_98_percentile = np.percentile(positive_values, 98)
    else:
        pos_98_percentile = None
        print("There is no value greater than 0")

    result = np.where(data > 0, data / pos_98_percentile, data / pos_98_percentile)

    return result

def gridnrom(data):
    mean = np.mean(data)
    std = np.std(data)

    min_value = mean - 3 * std
    max_value = mean + 3 * std
    return (data - min_value) / (max_value - min_value)

def z_score_normalize(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

class Pair_Loader_Nat(Dataset):
    def __init__(self):
        self.paths = glob("path to eeg encoding in training set")

        self.remove_indices = [2274, 2275]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        num = path.split("/")[-1].split("_")[-1].split(".")[0]
        name = path.split("/")[-1].split(f"_{num}.npy")[0]
        eeg = np.load(f"path/to/training_set/eeg_encoding/{name}_{num}.npy")
        eeg = np.delete(eeg, self.remove_indices, axis=0)
        whole_fmri = np.load(f"path/to/training_set/fmri_bold/{name}.npy")
        fmri = whole_fmri[:, int(num)]
        fmri = (fmri / 5 + 1) / 2
        fmri = np.delete(fmri, self.remove_indices, axis=0)

        return {'eeg': eeg, 'fmri':fmri}

