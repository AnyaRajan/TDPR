import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from sklearn.preprocessing import MinMaxScaler
from data_util import *  # Ensure get_augmentation_pipeline() and other helpers are defined here.
from features import *  # Ensure all feature extraction functions are defined here.
from omegaconf import OmegaConf
import models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

# --- Augmentation and Forward Pass Functions ---
def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def calculate_margin(pros):
    margins = []
    for sample_probs in pros:
        last_aug_probs = sample_probs[-1]  # Use final augmentation
        sorted_probs = np.sort(last_aug_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]
        margins.append(margin)
    return np.array(margins)

def calculate_mutual_information(pros):
    mi_scores = []
    for sample_probs in pros:
        avg_probs = np.mean(sample_probs, axis=0)
        entropy_avg = -np.sum(avg_probs * np.log(avg_probs + 1e-12))
        avg_entropy = np.mean([-np.sum(p * np.log(p + 1e-12)) for p in sample_probs])
        mi = entropy_avg - avg_entropy
        mi_scores.append(mi)
    return np.array(mi_scores)

# --- Helper Functions for Feature Extraction ---
def calculate_avg_pro_diff(pros):
    from sklearn.metrics.pairwise import cosine_similarity
    num_samples, num_aug, _ = pros.shape
    avg_diffs = np.zeros(num_samples)
    for i in range(num_samples):
        ref = pros[i, -1, :].reshape(1, -1)
        sims = cosine_similarity(pros[i, :num_aug-1, :], ref)
        distances = 1 - sims.flatten()
        avg_diffs[i] = np.mean(distances)
    return avg_diffs

def get_num_of_most_diff_class(labels):
    num_samples, num_aug = labels.shape
    max_diff = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        target = labels[i, -1]
        diff_counts = {}
        for j in range(num_aug - 1):
            if labels[i, j] != target:
                diff_counts[labels[i, j]] = diff_counts.get(labels[i, j], 0) + 1
        max_diff[i] = max(diff_counts.values()) if diff_counts else 0
    return max_diff

def calculate_kl_divergence(pros):
    kl_div = []
    for p in pros:
        base = p[-1]
        kl = [np.sum(p_i * np.log((p_i + 1e-12) / (base + 1e-12))) for p_i in p[:-1]]
        kl_div.append(np.mean(kl))
    return np.array(kl_div)

def calculate_agreement(labels):
    agreement_scores = []
    for row in labels:
        mode_label = np.bincount(row[:-1].astype(int)).argmax()
        agreement = np.sum(row[:-1] == mode_label) / (len(row) - 1)
        agreement_scores.append(agreement)
    return np.array(agreement_scores)

def calculate_confidence(pros):
    return np.max(pros[:, -1, :], axis=1)


def calculate_info_entropy(pros):
    entropys = []
    for pro in pros:
        entropy = -np.sum(pro * np.log2(pro))
        entropys.append(entropy)
    return entropys


