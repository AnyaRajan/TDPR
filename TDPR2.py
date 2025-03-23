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
from data_util import *  # Ensure this file defines get_augmentation_pipeline() and any other helpers you use.
from omegaconf import OmegaConf
import models

conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image

# --- Augmentation and Forward Pass Functions ---

def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def forward_with_augmentations(net, sample, num_aug=50):
    """ Generates `num_aug` augmented versions of `sample` and computes outputs. """
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
    """ Runs forward pass on all samples with augmentations and collects results. """
    num_samples = len(dataset)
    first_sample, _ = dataset[0]
    num_classes = net(torch.unsqueeze(first_sample.to(device), 0)).shape[1]  # Infer number of classes.

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

# --- Feature Extraction (using augmented outputs) ---
def extract_features(pros, labels, infos):
    """
    Extracts features from augmented outputs.
    - `pros`: (num_samples, num_aug, num_classes)
    - `labels`: (num_samples, num_aug)
    - `infos`: (num_samples, num_aug)
    """
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = np.mean(infos, axis=1)
    std_info = np.std(infos, axis=1)
    std_label = np.std(labels, axis=1)
    max_diff_num = get_num_of_most_diff_class(labels)

    feature = np.column_stack((std_label, avg_info, std_info, max_diff_num, avg_p_diff))
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

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

# --- Training Function ---
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

    # Train the model (no snapshots saved).
    train(net, conf.epochs, optimizer, criterion, trainloader)

    # Get validation and test DataLoaders.
    valloader, testloader = get_val_and_test(conf.corruption)

    # Generate augmented outputs for validation and test datasets.
    val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=50)
    test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=50)

    # Extract features from augmented outputs.
    val_features = extract_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    test_features = extract_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)

    print("Extracted validation features shape:", val_features.shape)
    print("Extracted test features shape:", test_features.shape)

    # Obtain error indices using standard test function.
    _, _, _, val_error_index = test(net, valloader)
    _, _, _, test_error_index = test(net, testloader)

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

    # Evaluate ranking using RAUC and ATRC metrics.
    print(rauc(is_bug, 100))
    print(rauc(is_bug, 200))
    print(rauc(is_bug, 500))
    print(rauc(is_bug, 1000))
    print(rauc(is_bug, test_num))
    print(ATRC(is_bug, len(test_error_index)))

if __name__ == '__main__':
    main()
