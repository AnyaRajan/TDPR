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
from data_util import *  # Ensure get_augmentation_pipeline() and other helpers are defined here
from omegaconf import OmegaConf
from PIL import Image

conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Augmentation and Forward Pass Functions ---
def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def forward_with_augmentations(net, sample, num_aug=100):
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

def generate_augmented_outputs(net, dataset, num_aug=100):
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
        entropy = -np.sum(pro * np.log2(pro + 1e-12))
        entropys.append(entropy)
    return entropys

# --- Test Function ---
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
    print(f"\n🧪 Final Test Accuracy: {acc:.2f}%")
    return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

# --- Training Function ---
def train(net, num_epochs, optimizer, criterion, trainloader, device):
    net.to(device)
    
    for epoch in range(num_epochs):
        net.train()  # Set model to training mode
        correct = 0
        total = 0
        running_loss = 0.0
        
        print(f"\nEpoch: {epoch + 1}")
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Compute training accuracy
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item()
        
        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)

        print(f"✅ Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# --- Neural Network for Ranking ---
class RankNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Single score output for ranking
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here - we want raw scores
        return x
    
    def predict(self, x):
        """Get ranking scores for samples"""
        self.eval()
        with torch.no_grad():
            return self.forward(x).squeeze().cpu().numpy()

# --- Pairwise Ranking Loss ---
def pairwise_ranking_loss(scores, labels):
    # Create all possible pairs of samples
    n_samples = scores.size(0)
    pairs = []
    targets = []
    
    # For each pair, determine which should rank higher
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if labels[i] != labels[j]:
                pairs.append((i, j))
                # Target is 1 if i should rank higher than j
                targets.append(1.0 if labels[i] > labels[j] else -1.0)
    
    if not pairs:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)
        
    pairs = torch.tensor(pairs, device=scores.device)
    targets = torch.tensor(targets, device=scores.device)
    
    # Get scores for the pairs
    s_i = scores[pairs[:, 0]]
    s_j = scores[pairs[:, 1]]
    
    # Calculate pairwise differences
    diff = s_i - s_j
    
    # Use a margin ranking loss (similar to RankNet approach)
    loss = F.logsigmoid(targets * diff).mean().neg()
    return loss

# --- Training function for the ranking network ---
def train_rank_net(model, features, labels, device, epochs=50, lr=0.001, batch_size=128):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    
    # Create dataset and dataloader for batch training
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            scores = model(batch_x).squeeze()
            
            # Calculate ranking loss
            loss = pairwise_ranking_loss(scores, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return model

# --- Function to compute Relative Area Under Curve ---
def rauc(is_bug, k):
    """
    Compute the relative area under the curve up to position k
    """
    if k > len(is_bug):
        k = len(is_bug)
    
    # Cumulative sum of bugs found
    cum_sum = np.cumsum(is_bug[:k])
    total_bugs = np.sum(is_bug)
    
    # Normalize by the maximum possible value
    max_area = min(total_bugs, k) * k - (min(total_bugs, k) * (min(total_bugs, k) - 1)) / 2
    
    # Compute the actual area
    actual_area = np.sum(cum_sum)
    
    return actual_area / max_area if max_area > 0 else 1.0

# --- Function to compute Average Time to Reveal Critical bugs ---
def ATRC(is_bug, total_bugs):
    """
    Compute the average time (position) to reveal critical bugs
    """
    bug_positions = np.where(is_bug == 1)[0] + 1  # +1 because positions are 1-indexed
    
    if len(bug_positions) == 0:
        return float('inf')
    
    return np.mean(bug_positions)

# --- Main Function ---
def main():
    # Initialize model and training data
    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    
    if conf.dataset in ["cifar10", "imagenet"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-4)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        
    # Train the model
    train(net, conf.epochs, optimizer, criterion, trainloader, device)
    
    # Get validation and test DataLoaders
    valloader, testloader = get_val_and_test(conf.corruption)
    
    # Set up file paths for saving intermediate arrays
    aug_file = "augmented_outputs.npz"
    feat_file = "extracted_features.npz"
    err_file = "error_indices.npz"
    
    # If augmented outputs exist, load them; otherwise compute and save
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
        val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=100)
        test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=100)
        np.savez(aug_file,
                 val_prob_arrays=val_prob_arrays,
                 val_label_arrays=val_label_arrays,
                 val_uncertainty_arrays=val_uncertainty_arrays,
                 test_prob_arrays=test_prob_arrays,
                 test_label_arrays=test_label_arrays,
                 test_uncertainty_arrays=test_uncertainty_arrays)
        print("Computed and saved augmented outputs.")
    
    # If extracted features exist, load them; otherwise compute and save
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
    
    # If error indices exist, load them; otherwise compute and save
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
    
    # Create binary labels: 0 for correct predictions, 1 for bug-revealing samples
    val_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_labels[val_error_index] = 1

    # Initialize and train the RankNet model
    print("\n🔄 Training RankNet for bug detection...")
    rank_model = RankNet(input_dim=val_features.shape[1])
    rank_model = train_rank_net(rank_model, val_features, val_labels, device, epochs=50)
    
    # Get predicted scores for test set
    X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
    scores = rank_model.predict(X_test)
    
    # Evaluate the ranking
    test_num = len(testloader.dataset)
    is_bug = np.zeros(test_num)
    is_bug[test_error_index] = 1
    index = np.argsort(scores)[::-1]  # Sort in descending order of bug likelihood
    is_bug = is_bug[index]
    
    print("\n📊 Ranking Performance Evaluation:")
    print(f"RAUC@100: {rauc(is_bug, 100):.4f}")
    print(f"RAUC@200: {rauc(is_bug, 200):.4f}")
    print(f"RAUC@500: {rauc(is_bug, 500):.4f}")
    print(f"RAUC@1000: {rauc(is_bug, 1000):.4f}")
    print(f"RAUC@all: {rauc(is_bug, test_num):.4f}")
    print(f"ATRC: {ATRC(is_bug, len(test_error_index)):.2f}")

if __name__ == '__main__':
    main()
