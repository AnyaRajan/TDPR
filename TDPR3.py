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
    
    print("Augmented outputs shape:", 
          all_prob_arrays.shape, all_label_arrays.shape, all_uncertainty_arrays.shape)
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
    
    # Debug: print shapes of all feature arrays
    print("Shape of std_label:", std_label.shape)
    print("Shape of avg_info:", avg_info.shape)
    print("Shape of std_info:", std_info.shape)
    print("Shape of max_diff_num:", max_diff_num.shape)
    print("Shape of avg_p_diff:", avg_p_diff.shape)
    
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
    all_probs, all_labels, all_infos = [], [], []
    error_indices = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            batch_size = inputs.size(0)
            
            all_probs.extend(probs)
            # Calculate information entropy for each sample in the batch
            batch_infos = calculate_info_entropy(probs)
            all_infos.extend(batch_infos)
            
            _, predicted = outputs.max(1)
            all_labels.extend(predicted.cpu().numpy())
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Compute incorrect mask and absolute indices for this batch
            incorrect_mask = ~predicted.eq(targets)
            if incorrect_mask.any():
                # Get the indices (relative to this batch) where predictions are incorrect.
                indices = torch.nonzero(incorrect_mask, as_tuple=False).view(-1).cpu().numpy()
                # Convert these to absolute indices within the dataset.
                absolute_indices = indices + batch_idx * batch_size
                error_indices.extend(absolute_indices.tolist())
    
    acc = 100. * correct / total
    return np.array(all_probs), np.array(all_labels), np.array(all_infos), np.array(error_indices)

# ... later in your main function when computing RAUC:

# Assume test_num is the number of samples in the test set
test_num = len(testloader.dataset)
# Run test() on the testloader to get test_error_index specific to test data
_, _, _, test_error_index = test(net, testloader)

is_bug = np.zeros(test_num)
is_bug[test_error_index] = 1

    return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

# --- Training Function (without snapshot saving) ---
def train(net, num_epochs, optimizer, criterion, trainloader):
    net.train()
    for epoch in range(num_epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0  
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:  # Print every 50 batches
                print(f"  🔹 Batch {batch_idx}: Loss = {loss.item():.4f}")

        print(f"✅ Epoch {epoch + 1} Finished. Avg Loss: {total_loss / len(trainloader):.4f}")

def main():
    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    
    if conf.dataset in ["cifar10", "imagenet"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), weight_decay=5e-4, momentum=0.9, lr=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
    print("🚀 Starting Training")
    train(net, conf.epochs, optimizer, criterion, trainloader)

    valloader, testloader = get_val_and_test(conf.corruption)

    print("⚡ Generating Augmented Outputs for Validation...")
    val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=50)
    
    # Instead of calling check_probability_variability again, use the generated outputs:
    first_sample_probs = val_prob_arrays[0]
    variance = np.var(first_sample_probs, axis=0)
    print("🔍 Probability Variance per Class (for first validation sample):", variance)
    print("🔍 Overall Variance (mean):", np.mean(variance))
    
    print("📊 Sample Augmented Probabilities (First 5):", val_prob_arrays[:5, -1, :])
    print("📊 Sample Uncertainty (First 5):", val_uncertainty_arrays[:5])

    val_features = extract_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    print("🛠 Extracted Feature Shape:", val_features.shape)
    print("📉 Feature Mean & Std:", np.mean(val_features, axis=0), np.std(val_features, axis=0))

    _, _, _, val_error_index = test(net, valloader)

    val_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_labels[val_error_index] = 1
    print(f"⚠️ {np.sum(val_labels)} misclassified samples out of {len(val_labels)}")

    print("⚡ Training XGBoost Ranking Model...")
    train_data = DMatrix(val_features, label=val_labels)
    xgb_rank_params = {'objective': 'rank:pairwise', 'max_depth': 5, 'learning_rate': 0.05}

    rankModel = xgboost.train(xgb_rank_params, train_data)

    print("🔮 Predicting Test Scores...")
    test_features = extract_features(*generate_augmented_outputs(net, testloader.dataset, num_aug=50))
    test_data = DMatrix(test_features)
    scores = rankModel.predict(test_data)

    print("🎯 Sample Scores (First 10):", scores[:10])

    test_num = len(testloader.dataset)
    _, _, _, test_error_index = test(net, testloader)
    is_bug = np.zeros(test_num)
    is_bug[test_error_index] = 1
    index = np.argsort(scores)[::-1]
    is_bug = is_bug[index]

    print("📊 RAUC and ATRC Metrics:")
    print("RAUC@100:", rauc(is_bug, 100))
    print("RAUC@200:", rauc(is_bug, 200))
    print("RAUC@500:", rauc(is_bug, 500))
    print("RAUC@1000:", rauc(is_bug, 1000))
    print("RAUC@all:", rauc(is_bug, test_num))
    print("ATRC:", ATRC(is_bug, len(test_error_index)))

if __name__ == '__main__':
    main()
