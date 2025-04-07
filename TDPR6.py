# --- Enhanced Ensemble Bug Detection for TPU ---
import os
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- TPU Setup ---
def setup_tpu():
    """Set up TPU for PyTorch acceleration"""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        
        device = xm.xla_device()
        print(f"TPU device detected: {device}")
        return device, True
    except ImportError:
        print("No TPU detected, falling back to GPU/CPU")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, False

device, is_tpu = setup_tpu()

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

# --- Optimized Model Architecture ---
class CIFAR10BugNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(CIFAR10BugNet, self).__init__()
        
        # Simplified architecture for TPU efficiency
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, x):
        features = self.features(x)
        attention = self.attention(features)
        weighted_features = features * attention
        return self.classifier(weighted_features)

# --- TPU-optimized training loop ---
def train_model_tpu(model, train_loader, val_loader, criterion, optimizer, scheduler, device, is_tpu, num_epochs=100):
    """Optimized training function for TPU"""
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_dict = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        if is_tpu:
            # Use parallel loader for TPU
            train_device_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
            for inputs, targets in train_device_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                xm.optimizer_step(optimizer)  # TPU-optimized gradient step
                train_loss += loss.item()
        else:
            # Regular training loop for GPU/CPU
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            if is_tpu:
                val_device_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
                for inputs, targets in val_device_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            else:
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

# --- Main Function for Ensemble Bug Detection ---
def run_ensemble_bug_detection(net, valloader, testloader, device, is_tpu):
    """Main function for running ensemble bug detection with TPU optimization"""
    # Extract ground truth errors
    _, _, _, val_error_index = test(net, valloader, device, is_tpu)
    _, _, _, test_error_index = test(net, testloader, device, is_tpu)
    
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
    
    # Create different feature subsets more efficiently
    print("Extracting features...")
    feature_set1 = extract_enhanced_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    feature_set2 = extract_original_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    feature_set3 = extract_statistical_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    
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
    for idx, features in enumerate(feature_sets):
        print(f"\nTraining model {idx+1} with feature set of shape {features.shape}")
        X_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(val_labels, dtype=torch.long)
        
        # Split validation data
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Create data loaders with larger batch size for TPU
        batch_size = 128 if is_tpu else 64
        train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
        val_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=is_tpu)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=is_tpu)
        
        # Initialize model
        model = CIFAR10BugNet(input_dim=features.shape[1]).to(device)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Train model with TPU-optimized code
        model = train_model_tpu(model, train_loader, val_loader, criterion, optimizer, scheduler, device, is_tpu)
        models.append(model)
    
    # Create test versions of each feature set
    print("Extracting test features...")
    test_feature_set1 = extract_enhanced_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    test_feature_set2 = extract_original_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    test_feature_set3 = extract_statistical_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    
    # Get ground truth for test dataset
    test_true_labels = np.zeros(len(testloader.dataset), dtype=int)
    for i, (_, target) in enumerate(testloader.dataset):
        test_true_labels[i] = target
    
    # Compute ensemble predictions in batches
    print("Computing ensemble predictions...")
    ensemble_scores = np.zeros(len(test_feature_set1))
    batch_size = 256  # Larger batch size for inference
    
    for idx, (model, test_features) in enumerate(zip(models, [test_feature_set1, test_feature_set2, test_feature_set3])):
        model.eval()
        # Process in batches to avoid memory issues
        for start_idx in range(0, len(test_features), batch_size):
            end_idx = min(start_idx + batch_size, len(test_features))
            X_test_batch = torch.tensor(test_features[start_idx:end_idx], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits = model(X_test_batch)
                probs = F.softmax(logits, dim=1)
                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    scores_batch = probs[:, 1].cpu().numpy()
                else:
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
    
    # Skip class-specific models for TPU efficiency
    print("Skipping class-specific models for TPU efficiency")
    
    # Sort and evaluate
    test_num = len(testloader.dataset)
    is_bug = np.zeros(test_num)
    is_bug[test_error_index] = 1
    index = np.argsort(ensemble_scores)[::-1]
    is_bug = is_bug[index]
    
    print("\n--- Final Evaluation with TPU-Optimized Models ---")
    print("RAUC@100:", rauc(is_bug, 100))
    print("RAUC@200:", rauc(is_bug, 200))
    print("RAUC@500:", rauc(is_bug, 500))
    print("RAUC@1000:", rauc(is_bug, 1000))
    print("RAUC@all:", rauc(is_bug, test_num))
    print("ATRC:", ATRC(is_bug, len(test_error_index)))
    
    return ensemble_scores, is_bug

# --- Test function (needed for completeness) ---
def test(net, testloader, device, is_tpu):
    """Test function that returns error indices"""
    # Placeholder implementation - replace with your actual testing function
    error_index = []  # This should be implemented based on your original code
    return None, None, None, error_index

# --- Main function ---
def main():
    # Initialize model and training data
    device, is_tpu = setup_tpu()
    
    # Your model initialization code here
    # net = models.__dict__[conf.model]().to(device)
    # trainloader = get_train_data(conf.dataset)
    
    # Your training code here
    # train(net, conf.epochs, optimizer, criterion, trainloader, device, is_tpu)
    
    # Run ensemble bug detection
    print("\n--- Running TPU-Optimized Ensemble Bug Detection ---")
    # final_scores, is_bug = run_ensemble_bug_detection(net, valloader, testloader, device, is_tpu)
    
if __name__ == '__main__':
    main()
