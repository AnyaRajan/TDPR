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
from scipy import stats
from data_util import *  # Ensure get_augmentation_pipeline() and other helpers are defined here.
from omegaconf import OmegaConf
import models
from PIL import Image

conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def forward_with_augmentations(net, sample, num_aug=200):  # Increased from 100 to 200
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

def generate_augmented_outputs(net, dataset, num_aug=200):  # Increased from 100 to 200
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

def calculate_variance_of_top_k_probs(pros, k=3):
    """Calculate variance of top-k probability values across augmentations"""
    num_samples, num_aug, num_classes = pros.shape
    var_top_k = np.zeros(num_samples)
    
    for i in range(num_samples):
        top_k_values = np.sort(pros[i], axis=1)[:, -k:]  # Get top-k values for each augmentation
        var_top_k[i] = np.mean(np.var(top_k_values, axis=0))  # Average variance across top-k positions
        
    return var_top_k

def calculate_prediction_flip_rate(labels):
    """Calculate how frequently the prediction changes across augmentations"""
    num_samples, num_aug = labels.shape
    flip_rates = np.zeros(num_samples)
    
    for i in range(num_samples):
        flips = np.sum(labels[i, 1:] != labels[i, :-1])
        flip_rates[i] = flips / (num_aug - 1)
        
    return flip_rates

def extract_enhanced_features(pros, labels, infos):
    """Extract enhanced features for bug detection"""
    # Import needed modules
    from sklearn.decomposition import PCA
    
    # Current features
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = np.mean(infos, axis=1)
    std_info = np.std(infos, axis=1)
    std_label = np.std(labels, axis=1)
    max_diff_num = get_num_of_most_diff_class(labels)
    
    # Top prediction probabilities
    mean_top_prob = np.mean(np.max(pros, axis=2), axis=1)
    std_top_prob = np.std(np.max(pros, axis=2), axis=1)
    
    # Top-2 probability difference (margin between most confident and second most confident class)
    sorted_probs = np.sort(pros, axis=2)[:, :, -2:]  # Get top 2 probs for each augmentation
    margin = sorted_probs[:, :, 1] - sorted_probs[:, :, 0]  # Difference between top 2
    avg_margin = np.mean(margin, axis=1)
    std_margin = np.std(margin, axis=1)
    min_margin = np.min(margin, axis=1)  # Minimum margin is often indicative of potential bugs
    
    # Consistency of top prediction across augmentations
    modal_class = stats.mode(labels, axis=1)[0].flatten()
    consistency = np.array([np.sum(labels[i] == modal_class[i]) / labels.shape[1] for i in range(labels.shape[0])])
    
    # New feature: Prediction flip rate
    flip_rate = calculate_prediction_flip_rate(labels)
    
    # New feature: Variance of top-k probabilities
    var_top_3 = calculate_variance_of_top_k_probs(pros, k=3)
    
    # Class-specific confidence statistics
    num_classes = pros.shape[2]
    class_conf_stats = []
    for i in range(num_classes):
        class_mask = (labels == i)
        class_conf = np.zeros(pros.shape[0])
        for j in range(pros.shape[0]):
            if np.any(class_mask[j]):
                class_conf[j] = np.mean(np.max(pros[j, class_mask[j]], axis=1))
            else:
                # If no augmentations predict this class, use the mean probability for this class
                class_conf[j] = np.mean(pros[j, :, i])
        class_conf_stats.append(class_conf)
    class_conf_stats = np.column_stack(class_conf_stats)
    
    # Additional feature: Entropy variance across different augmentations
    entropy_var = np.var(infos, axis=1)
    
    # Additional feature: Maximum entropy
    max_entropy = np.max(infos, axis=1)
    
    # Combine all features
    feature = np.column_stack((
        std_label, avg_info, std_info, entropy_var, max_entropy, 
        max_diff_num, avg_p_diff, avg_margin, std_margin, min_margin,
        mean_top_prob, std_top_prob, consistency, flip_rate, var_top_3,
        class_conf_stats
    ))
    
    # Apply dimensionality reduction if feature space is too large
    if feature.shape[1] > 50:
        pca = PCA(n_components=min(feature.shape[0], 50))
        feature = pca.fit_transform(feature)
    
    # Normalize features
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

# Use the enhanced feature extraction function as extract_features for compatibility
extract_features = extract_enhanced_features

def calculate_info_entropy(pros):
    entropys = []
    for pro in pros:
        entropy = -np.sum(pro * np.log2(pro + 1e-12))  # Added epsilon to avoid log(0)
        entropys.append(entropy)
    return entropys

# --- Test Function (improved) ---
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

# --- Training Function (improved) ---
def train(net, num_epochs, optimizer, criterion, trainloader, device):
    net.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    best_loss = float('inf')
    
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
        
        # Update learning rate based on validation loss
        scheduler.step(avg_loss)

        print(f"✅ Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), 'best_model.pth')
            print("Model saved!")

# --- Updated BugNet (with improved architecture) ---
class BugNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(BugNet, self).__init__()
        
        # Feature processing layers with wider network and stronger regularization
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 2)
        )
        
    def forward(self, x):
        features = self.feature_layers(x)
        
        # Apply attention
        attention_weights = torch.sigmoid(self.attention(features))
        weighted_features = features * attention_weights
        
        return self.classifier(weighted_features)

# --- Main Function (improved) ---
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
    train(net, conf.epochs, optimizer, criterion, trainloader, device)
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
        val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=200)
        test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=200)
        np.savez(aug_file,
                 val_prob_arrays=val_prob_arrays,
                 val_label_arrays=val_label_arrays,
                 val_uncertainty_arrays=val_uncertainty_arrays,
                 test_prob_arrays=test_prob_arrays,
                 test_label_arrays=test_label_arrays,
                 test_uncertainty_arrays=test_uncertainty_arrays)
        print("Computed and saved augmented outputs.")
    
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

    # Handle class imbalance with weighted loss
    pos_weight = len(val_labels) / max(1, np.sum(val_labels))
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Handle imbalanced dataset using class weights
    num_pos = np.sum(val_labels)
    num_neg = len(val_labels) - num_pos
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}")
    
    # Use a larger hidden dimension for the bug detector
    hidden_dim = 128  # Increased from 64
    
    # Convert data to PyTorch tensors
    X_train = torch.tensor(val_features, dtype=torch.float32)
    y_train = torch.tensor(val_labels, dtype=torch.long)
    X_test = torch.tensor(test_features, dtype=torch.float32)
    
    # Define model, loss, optimizer
    model = BugNet(input_dim=val_features.shape[1], hidden_dim=hidden_dim).to(device)
    
    # Use weighted cross-entropy loss to handle class imbalance
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use AdamW optimizer with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop with validation
    best_val_loss = float('inf')
    patience = 10  # early stopping patience
    patience_counter = 0
    
    # Split validation set into training and validation
    val_size = int(0.8 * len(X_train))
    train_indices = np.random.choice(len(X_train), size=val_size, replace=False)
    val_indices = np.array([i for i in range(len(X_train)) if i not in train_indices])
    
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val_split = X_train[val_indices]
    y_val_split = y_train[val_indices]
    
    print(f"Training on {len(X_train_split)} samples, validating on {len(X_val_split)} samples")
    
    # Create DataLoader for mini-batch training
    train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    
    num_epochs = 100  # Increased from 50
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_split.to(device))
            val_loss = criterion(val_outputs, y_val_split.to(device))
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct = (val_predicted == y_val_split.to(device)).sum().item()
            val_acc = 100.0 * val_correct / len(y_val_split)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Early stopping and model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_bugnet.pth')
            print("Saved best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_bugnet.pth'))
    
    # Get predicted probabilities for test set
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        probs = F.softmax(logits, dim=1)
        scores = probs[:, 1].cpu().numpy()  # Take probability of class '1' (bug)
    
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
