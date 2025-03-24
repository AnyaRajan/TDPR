import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from sklearn.preprocessing import MinMaxScaler
from xgboost import DMatrix
import xgboost
from data_util import *  # Ensure get_augmentation_pipeline() and other helpers are defined here.
from omegaconf import OmegaConf
import models
from PIL import Image

conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Augmentation and Forward Pass Functions ---
def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def forward_with_augmentations(net, sample, num_aug=50):
    if isinstance(sample, torch.Tensor):
        sample = transforms.ToPILImage()(sample.cpu())
    aug_pipeline = get_augmentation_pipeline()  # Defined in data_util.py
    prob_list, label_list, uncertainty_list = [], [], []
    net.eval()
    with torch.no_grad():
        for _ in range(num_aug):
            aug_sample = aug_pipeline(sample).unsqueeze(0).to(device)
            outputs = net(aug_sample)
            probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze(0)
            prob_list.append(probs)
            label_list.append(np.argmax(probs))
            uncertainty_list.append(calculate_info_entropy_from_probs(probs))
    return np.array(prob_list), np.array(label_list), np.array(uncertainty_list)

def generate_augmented_outputs(net, dataset, num_aug=50):
    num_samples = len(dataset)
    first_sample, _ = dataset[0]
    if isinstance(first_sample, torch.Tensor):
        first_sample = first_sample.to(device)
    with torch.no_grad():
        num_classes = net(first_sample.unsqueeze(0)).shape[1]
    
    all_prob_arrays = np.zeros((num_samples, num_aug, num_classes))
    all_label_arrays = np.zeros((num_samples, num_aug))
    all_uncertainty_arrays = np.zeros((num_samples, num_aug))
    
    for idx in range(num_samples):
        sample, _ = dataset[idx]
        probs, labels, uncertainties = forward_with_augmentations(net, sample, num_aug=num_aug)
        all_prob_arrays[idx] = probs
        all_label_arrays[idx] = labels
        all_uncertainty_arrays[idx] = uncertainties
    return all_prob_arrays, all_label_arrays, all_uncertainty_arrays

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

def extract_features(pros, labels, infos):
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = np.mean(infos, axis=1)
    std_info = np.std(infos, axis=1)
    std_label = np.std(labels, axis=1)
    max_diff_num = get_num_of_most_diff_class(labels)
    feature = np.column_stack((std_label, avg_info, std_info, max_diff_num, avg_p_diff))
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

def calculate_info_entropy(pros):
    entropys = []
    for pro in pros:
        entropy = -np.sum(pro * np.log2(pro))
        entropys.append(entropy)
    return entropys

# --- Original Test Function (unchanged) ---
def test(net, testloader):
    net.eval()
    correct, total = 0, 0
    pros, labels, infos, error_index = [], [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            pro = F.softmax(outputs, dim=1).cpu().numpy()
            pros.extend(pro)
            infos.extend(calculate_info_entropy(pro))
            _, predicted = outputs.max(1)
            labels.extend(predicted.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            incorrect_mask = ~predicted.eq(targets)
            if incorrect_mask.any():
                incorrect_indices = (batch_idx * testloader.batch_size) + torch.nonzero(incorrect_mask).view(-1)
                error_index.extend(incorrect_indices.tolist())
    acc = 100. * correct / total
    return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

# --- Training Function (without snapshot saving) ---
def train(net, num_epochs, optimizer, criterion, trainloader):
    net.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# --- Main Function ---
def main():
    # Initialize model and training data.
    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    if conf.dataset in ["cifar10", "imagenet"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), weight_decay=5e-4, momentum=0.9, lr=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
    # Train the model.
    print("🚀 Starting Training")
    train(net, conf.epochs, optimizer, criterion, trainloader)
    
    # Get validation and test DataLoaders.
    valloader, testloader = get_val_and_test(conf.corruption)
    
    # Set up file paths for saving intermediate arrays.
    aug_file = "augmented_outputs.npz"
    feat_file = "extracted_features.npz"
    err_file = "error_indices.npz"
    
    # If augmented outputs exist, load them; otherwise compute and save.
    if os.path.exists(aug_file):
        data = np.load(aug_file)
        val_prob_arrays = data['val_prob_arrays']
        val_label_arrays = data['val_label_arrays']
        val_uncertainty_arrays = data['val_uncertainty_arrays']
        test_prob_arrays = data['test_prob_arrays']
        test_label_arrays = data['test_label_arrays']
        test_uncertainty_arrays = data['test_uncertainty_arrays']
        print("Loaded augmented outputs from file.")
    else:
        val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=50)
        test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=50)
        np.savez(aug_file,
                 val_prob_arrays=val_prob_arrays,
                 val_label_arrays=val_label_arrays,
                 val_uncertainty_arrays=val_uncertainty_arrays,
                 test_prob_arrays=test_prob_arrays,
                 test_label_arrays=test_label_arrays,
                 test_uncertainty_arrays=test_uncertainty_arrays)
        print("Computed and saved augmented outputs.")
    
    # Check probability variability using the first validation sample's augmented outputs.
    first_sample_probs = val_prob_arrays[0]  # shape: (num_aug, num_classes)
    variance = np.var(first_sample_probs, axis=0)
    print("🔍 Probability Variance per Class (for first validation sample):", variance)
    print("🔍 Overall Variance (mean):", np.mean(variance))
    
    # If extracted features exist, load them; otherwise compute and save.
    if os.path.exists(feat_file):
        feat_data = np.load(feat_file)
        val_features = feat_data['val_features']
        test_features = feat_data['test_features']
        print("Loaded extracted features from file.")
    else:
        val_features = extract_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
        test_features = extract_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
        np.savez(feat_file, val_features=val_features, test_features=test_features)
        print("Computed and saved extracted features.")
    
    # If error indices exist, load them; otherwise compute and save.
    if os.path.exists(err_file):
        err_data = np.load(err_file)
        val_error_index = err_data['val_error_index']
        test_error_index = err_data['test_error_index']
        print("Loaded error indices from file.")
    else:
        _, _, _, val_error_index = test(net, valloader)
        _, _, _, test_error_index = test(net, testloader)
        np.savez(err_file, val_error_index=val_error_index, test_error_index=test_error_index)
        print("Computed and saved error indices.")
    
    print("Extracted validation features shape:", val_features.shape)
    print("Extracted test features shape:", test_features.shape)
    
    # Create binary labels: 0 for correct predictions, 1 for bug-revealing samples.
    val_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_labels[val_error_index] = 1

    # Build and train the ranking model using XGBoost.
    xgb_rank_params = {
        'objective': 'rank:pairwise',
        'colsample_bytree': 0.5,
        'nthread': -1,
        'eval_metric': 'ndcg',
        'max_depth': 5,
        'min_child_weight': 1,
        'learning_rate': 0.05,
    }
    train_data = DMatrix(val_features, label=val_labels)
    rankModel = xgboost.train(xgb_rank_params, train_data)
    
    # Predict scores on test features.
    test_data = DMatrix(test_features)
    scores = rankModel.predict(test_data)
    
    test_num = len(testloader.dataset)
    is_bug = np.zeros(test_num)
    is_bug[test_error_index] = 1
    index = np.argsort(scores)[::-1]
    is_bug = is_bug[index]
    
    print("RAUC@100:", rauc(is_bug, 100))
    print("RAUC@200:", rauc(is_bug, 200))
    print("RAUC@500:", rauc(is_bug, 500))
    print("RAUC@1000:", rauc(is_bug, 1000))
    print("RAUC@all:", rauc(is_bug, test_num))
    print("ATRC:", ATRC(is_bug, len(test_error_index)))

if __name__ == '__main__':
    main()
