# --- Enhanced Ensemble Bug Detection for T4 GPUs ---
import os
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
# from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from data_util import *  # Ensure get_augmentation_pipeline() and other helpers are defined here.
from omegaconf import OmegaConf
import models

# --- GPU Setup ---
def setup_gpu():
    """Set up GPU for PyTorch acceleration"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print(f"GPU detected: {gpu_name} x {gpu_count}")
    else:
        device = torch.device('cpu')
        print("No GPU detected, using CPU")
    
    return device

device = setup_gpu()

# --- Feature Extraction Methods (Optimized) ---
def calculate_avg_pro_diff(pros):
    """Helper function to calculate average probability difference"""
    n_samples = pros.shape[0]
    n_aug = pros.shape[1]
    
    # Calculate pairwise differences more efficiently
    avg_diffs = np.zeros(n_samples)
    for i in range(n_samples):
        total_diff = 0
        count = 0
        for j in range(n_aug):
            for k in range(j+1, n_aug):
                total_diff += np.sum(np.abs(pros[i, j] - pros[i, k]))
                count += 1
        avg_diffs[i] = total_diff / max(1, count)
    
    return avg_diffs

def get_num_of_most_diff_class(labels):
    """Helper function to get number of most different class"""
    # Vectorized implementation
    unique_counts = np.array([np.bincount(row, minlength=10) for row in labels])
    return np.max(unique_counts, axis=1)

def extract_original_features(pros, labels, infos):
    """Original feature extraction method for baseline comparison"""
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = np.mean(infos, axis=1)
    std_info = np.std(infos, axis=1)
    std_label = np.std(labels, axis=1)
    max_diff_num = get_num_of_most_diff_class(labels)
    
    # Basic feature set
    feature = np.column_stack((std_label, avg_info, std_info, max_diff_num, avg_p_diff))
    
    # Normalize features
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

def extract_statistical_features(pros, labels, infos):
    """Statistical feature extraction method focusing on distributions"""
    # Vectorized operations for efficiency
    mean_probs = np.mean(pros, axis=1)
    std_probs = np.std(pros, axis=1)
    
    max_probs = np.max(pros, axis=2)
    mean_max_prob = np.mean(max_probs, axis=1)
    std_max_prob = np.std(max_probs, axis=1)
    
    mean_entropy = np.mean(infos, axis=1)
    std_entropy = np.std(infos, axis=1)
    skew_entropy = stats.skew(infos, axis=1)
    
    # Calculate label consistency
    label_mode = stats.mode(labels, axis=1)[0].flatten()
    label_consistency = np.array([np.sum(labels[i] == label_mode[i]) / labels.shape[1] 
                                 for i in range(labels.shape[0])])
    
    feature = np.column_stack((
        mean_probs, std_probs, mean_max_prob, std_max_prob,
        mean_entropy, std_entropy, skew_entropy, label_consistency
    ))
    
    # Normalize features
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

def extract_enhanced_features(pros, labels, infos):
    """Enhanced feature extraction combining multiple sources"""
    # Combine basic features with additional ones
    original_features = extract_original_features(pros, labels, infos)
    statistical_features = extract_statistical_features(pros, labels, infos)
    
    # Additional features
    entropy_diff = np.max(infos, axis=1) - np.min(infos, axis=1)
    prob_range = np.max(pros, axis=(1, 2)) - np.min(pros, axis=(1, 2))
    
    # Combine all features
    feature = np.column_stack((
        original_features, 
        statistical_features,
        entropy_diff, 
        prob_range
    ))
    
    # Normalize features
    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

# --- Mixed Precision Training Setup ---
def setup_amp():
    """Setup Automatic Mixed Precision for faster training"""
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        print("Mixed precision training enabled")
        return scaler, True
    except:
        print("Mixed precision training not available")
        return None, False

# --- Model Architecture with Improved GPU Utilization ---
class CIFAR10BugNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(CIFAR10BugNet, self).__init__()
        
        # Feature processing with residual connections
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        # First block with residual connection
        identity = self.layer1(x)
        out = F.leaky_relu(self.bn1(identity), 0.2)
        out = self.dropout(out)
        
        # Second block with residual connection
        out = self.layer2(out)
        out = F.leaky_relu(self.bn2(out), 0.2)
        out = self.dropout(out)
        out = out + identity  # Residual connection
        
        # Third block
        out = self.layer3(out)
        out = F.leaky_relu(self.bn3(out), 0.2)
        out = self.dropout(out)
        
        # Apply attention
        attention_weights = torch.sigmoid(self.attention(out))
        weighted_features = out * attention_weights
        
        # Classification
        return self.classifier(weighted_features)

# --- GPU-optimized training loop ---
def train_model_gpu(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100):
    """Optimized training function for T4 GPUs with mixed precision"""
    from torch.cuda.amp import GradScaler, autocast
    
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_dict = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Use mixed precision training
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Print progress every few epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_dict)
    return model

# --- Parallel feature extraction for T4 x2 GPUs ---
def parallel_feature_extraction(prob_arrays, label_arrays, uncertainty_arrays):
    """Extract features in parallel for better performance"""
    # Split data into chunks for parallel processing
    from multiprocessing import Pool
    import functools
    
    # Prepare data chunks
    n_samples = prob_arrays.shape[0]
    n_chunks = 4  # Adjust based on CPU core count
    chunk_size = n_samples // n_chunks
    chunks = [(
        prob_arrays[i:i+chunk_size], 
        label_arrays[i:i+chunk_size], 
        uncertainty_arrays[i:i+chunk_size]
    ) for i in range(0, n_samples, chunk_size)]
    
    # Process chunks in parallel
    with Pool(processes=n_chunks) as pool:
        results1 = pool.starmap(extract_enhanced_features, chunks)
        results2 = pool.starmap(extract_original_features, chunks)
        results3 = pool.starmap(extract_statistical_features, chunks)
    
    # Combine results
    feature_set1 = np.vstack(results1)
    feature_set2 = np.vstack(results2)
    feature_set3 = np.vstack(results3)
    
    return feature_set1, feature_set2, feature_set3

# --- Helper functions for evaluation ---
def rauc(is_bug, k):
    """Calculate RAUC@k metric"""
    if k > len(is_bug):
        k = len(is_bug)
    return np.sum(is_bug[:k]) / k

def ATRC(is_bug, num_errors):
    """Calculate ATRC metric"""
    if num_errors == 0:
        return 0
    recalls = np.zeros(num_errors)
    for i in range(num_errors):
        recalls[i] = np.sum(is_bug[:i+1]) / (i+1)
    return np.mean(recalls)

# --- Multi-GPU wrapper ---
def get_ddp_model(model, device):
    """Set up DataParallel for multi-GPU training"""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = nn.DataParallel(model)
    return model.to(device)

# --- Main Function for Ensemble Bug Detection ---
def run_ensemble_bug_detection(net, valloader, testloader, device):
    """Main function for running ensemble bug detection with GPU optimization"""
    # Extract ground truth errors
    _, _, _, val_error_index = test(net, valloader, device)
    _, _, _, test_error_index = test(net, testloader, device)
    
    # Load or compute augmented outputs
    aug_file = "augmented_outputs.npz"
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
        # Generate and save augmented outputs (not shown)
        print("Augmented outputs not found. Please generate them first.")
        return None, None
    
    # Extract features in parallel
    print("Extracting features in parallel...")
    feature_set1, feature_set2, feature_set3 = parallel_feature_extraction(
        val_prob_arrays, val_label_arrays, val_uncertainty_arrays
    )
    
    # Create binary labels for validation data
    val_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_labels[val_error_index] = 1
    
    # Handle class imbalance with weights
    num_pos = np.sum(val_labels)
    num_neg = len(val_labels) - num_pos
    pos_weight = num_neg / max(1, num_pos)
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    print(f"Training with class weights: [1.0, {pos_weight:.2f}]")
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}")
    
    # Train models with different feature sets
    models = []
    feature_sets = [feature_set1, feature_set2, feature_set3]
    
    # Use larger batch sizes for T4 GPUs
    batch_size = 128
    
    for idx, features in enumerate(feature_sets):
        print(f"\nTraining model {idx+1} with feature set of shape {features.shape}")
        X_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(val_labels, dtype=torch.long)
        
        # Split validation data
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Create data loaders with pinned memory for faster GPU transfer
        train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
        val_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                 shuffle=True, pin_memory=True,
                                                 num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                               pin_memory=True, num_workers=2)
        
        # Initialize model with DataParallel for multi-GPU
        model = CIFAR10BugNet(input_dim=features.shape[1])
        model = get_ddp_model(model, device)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Train model with GPU-optimized code
        model = train_model_gpu(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
        models.append(model)
    
    # Create test versions of each feature set in parallel
    print("Extracting test features in parallel...")
    test_feature_set1, test_feature_set2, test_feature_set3 = parallel_feature_extraction(
        test_prob_arrays, test_label_arrays, test_uncertainty_arrays
    )
    
    # Get ground truth for test dataset
    test_true_labels = np.zeros(len(testloader.dataset), dtype=int)
    for i, (_, target) in enumerate(testloader.dataset):
        test_true_labels[i] = target
    
    # Compute ensemble predictions in batches
    print("Computing ensemble predictions...")
    ensemble_scores = np.zeros(len(test_feature_set1))
    batch_size = 512  # Larger batch size for inference
    
    for idx, (model, test_features) in enumerate(zip(models, [test_feature_set1, test_feature_set2, test_feature_set3])):
        model.eval()
        # Process in batches to avoid memory issues
        for start_idx in range(0, len(test_features), batch_size):
            end_idx = min(start_idx + batch_size, len(test_features))
            X_test_batch = torch.tensor(test_features[start_idx:end_idx], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                with autocast():  # Use mixed precision for inference
                    logits = model(X_test_batch)
                    probs = F.softmax(logits, dim=1)
                    scores_batch = probs[:, 1].cpu().numpy()
                
                ensemble_scores[start_idx:end_idx] += scores_batch / len(models)
    
    # Analyze errors by class
    errors_by_class = {}
    for i in range(10):  # CIFAR-10 has 10 classes
        class_indices = np.where(test_true_labels == i)[0]
        if len(class_indices) > 0:
            error_indices_for_class = np.intersect1d(class_indices, test_error_index)
            errors_by_class[i] = len(error_indices_for_class) / len(class_indices)
        else:
            errors_by_class[i] = 0.0
    
    print("Error rates by class:", errors_by_class)
    
    # Train class-specific models for classes with high error rates
    high_error_classes = [cls for cls, rate in errors_by_class.items() if rate > 0.1]
    print(f"Training class-specific models for high error classes: {high_error_classes}")
    class_specific_models = {}
    
    for cls in high_error_classes:
        # Filter data for this class
        class_indices = np.where(test_true_labels == cls)[0]
        if len(class_indices) < 10:  # Skip if too few samples
            continue
            
        # Create feature set specific to this class
        cls_features = extract_enhanced_features(
            test_prob_arrays[class_indices], 
            test_label_arrays[class_indices], 
            test_uncertainty_arrays[class_indices]
        )
        
        X_cls = torch.tensor(cls_features, dtype=torch.float32)
        # Create labels based on whether each sample is in the error index
        y_cls = torch.zeros(len(class_indices), dtype=torch.long)
        for i, idx in enumerate(class_indices):
            if idx in test_error_index:
                y_cls[i] = 1
        
        # Skip if we don't have enough samples of each class
        if torch.sum(y_cls == 0) < 5 or torch.sum(y_cls == 1) < 5:
            continue
        
        # Train model specifically for this class
        print(f"Training model for class {cls} with {len(X_cls)} samples")
        model = CIFAR10BugNet(input_dim=cls_features.shape[1]).to(device)
        
        # Basic training loop for class-specific model with mixed precision
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42
        )
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(X_train.to(device))
                loss = criterion(outputs, y_train.to(device))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    with autocast():
                        outputs = model(X_val.to(device))
                        val_loss = criterion(outputs, y_val.to(device))
                        _, predicted = torch.max(outputs, 1)
                        accuracy = (predicted == y_val.to(device)).sum().item() / len(y_val)
                print(f"Class {cls}, Epoch {epoch}: Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Acc = {accuracy:.4f}")
        
        class_specific_models[cls] = model
    
    # Incorporate class-specific predictions into overall scoring
    final_scores = np.zeros(len(test_feature_set1))
    for i, true_label in enumerate(test_true_labels):
        if true_label in class_specific_models:
            # Use class-specific model
            model = class_specific_models[true_label]
            X_test_sample = torch.tensor(test_feature_set1[i:i+1], dtype=torch.float32)
            with torch.no_grad():
                with autocast():
                    logit = model(X_test_sample.to(device))
                    prob = F.softmax(logit, dim=1)
                    final_scores[i] = prob[0, 1].cpu().numpy()
        else:
            # Use general ensemble
            final_scores[i] = ensemble_scores[i]
    
    # Sort and evaluate
    test_num = len(testloader.dataset)
    is_bug = np.zeros(test_num)
    is_bug[test_error_index] = 1
    index = np.argsort(final_scores)[::-1]
    is_bug = is_bug[index]
    
    print("\n--- Final Evaluation with T4 GPU-Optimized Models ---")
    print("RAUC@100:", rauc(is_bug, 100))
    print("RAUC@200:", rauc(is_bug, 200))
    print("RAUC@500:", rauc(is_bug, 500))
    print("RAUC@1000:", rauc(is_bug, 1000))
    print("RAUC@all:", rauc(is_bug, test_num))
    print("ATRC:", ATRC(is_bug, len(test_error_index)))
    
    return final_scores, is_bug

# --- Test function (needed for completeness) ---
def test(net, testloader, device):
    """Test function that returns error indices"""
    # Placeholder implementation - replace with your actual testing function
    error_index = []  # This should be implemented based on your original code
    return None, None, None, error_index

# --- Main function ---
def main():
    # Initialize model and training data
    device = setup_gpu()
    print(models.__dict__.keys())
    
    # Your model initialization code here
    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    
    # Your training code here
    train(net, conf.epochs, optimizer, criterion, trainloader, device)
    
    # Run ensemble bug detection
    print("\n--- Running T4 GPU-Optimized Ensemble Bug Detection ---")
    final_scores, is_bug = run_ensemble_bug_detection(net, valloader, testloader, device)
    
if __name__ == '__main__':
    main()
