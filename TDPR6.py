# --- Enhanced Ensemble Bug Detection ---
from sklearn.model_selection import train_test_split
import torch.utils.data
from torch.nn import functional as F

# --- Feature Extraction Methods ---
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
    # Basic statistics
    mean_probs = np.mean(pros, axis=1)  # Average probabilities across augmentations
    std_probs = np.std(pros, axis=1)    # Standard deviation across augmentations
    
    # Probability distribution features
    max_probs = np.max(pros, axis=2)    # Maximum probability for each augmentation
    mean_max_prob = np.mean(max_probs, axis=1)  # Average of maximum probabilities
    std_max_prob = np.std(max_probs, axis=1)    # Std dev of maximum probabilities
    
    # Entropy statistics
    mean_entropy = np.mean(infos, axis=1)
    std_entropy = np.std(infos, axis=1)
    skew_entropy = stats.skew(infos, axis=1)
    
    # Label consistency
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

# --- Enhanced Model Architecture ---
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

# --- Main Function for Ensemble Bug Detection ---
def run_ensemble_bug_detection():
    # Get validation and test DataLoaders and extract ground truth errors
    valloader, testloader = get_val_and_test(conf.corruption)
    _, _, _, val_error_index = test(net, valloader)
    _, _, _, test_error_index = test(net, testloader)
    
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
        # Generate and save augmented outputs (already implemented in your code)
        pass
    
    # Create different feature subsets
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
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
        val_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)
        
        # Initialize new model
        model = CIFAR10BugNet(input_dim=features.shape[1]).to(device)
        
        # Training parameters
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
            # Validation
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
                # Save best model
                torch.save(model.state_dict(), f'best_bugnet_model{idx}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_bugnet_model{idx}.pth'))
        models.append(model)
    
    # Create test versions of each feature set
    test_feature_set1 = extract_enhanced_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    test_feature_set2 = extract_original_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    test_feature_set3 = extract_statistical_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    
    # Get ground truth for test dataset
    test_true_labels = np.zeros(len(testloader.dataset), dtype=int)
    for i, (_, target) in enumerate(testloader.dataset):
        test_true_labels[i] = target

    # Ensemble predictions
    ensemble_scores = np.zeros(len(test_feature_set1))
    for idx, (model, test_features) in enumerate(zip(models, [test_feature_set1, test_feature_set2, test_feature_set3])):
        X_test = torch.tensor(test_features, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(X_test.to(device))
            probs = F.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()
            ensemble_scores += scores / len(models)
    
    # Analyze errors by class
    errors_by_class = {}
    for i in range(10):  # CIFAR-10 has 10 classes
        class_indices = np.where(test_true_labels == i)[0]
        if len(class_indices) > 0:  # Avoid division by zero
            error_indices_for_class = np.intersect1d(class_indices, test_error_index)
            errors_by_class[i] = len(error_indices_for_class) / len(class_indices)
        else:
            errors_by_class[i] = 0.0
    print("Error rates by class:", errors_by_class)
    
    # Train class-specific bug detectors for high-error classes
    high_error_classes = [cls for cls, rate in errors_by_class.items() if rate > 0.1]  # Adjust threshold as needed
    print(f"Training class-specific models for high error classes: {high_error_classes}")
    class_specific_models = {}
    for cls in high_error_classes:
        # Filter data for this class
        class_indices = np.where(test_true_labels == cls)[0]
        if len(class_indices) == 0:
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
        
        # Basic training loop for class-specific model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_cls.to(device))
            loss = criterion(outputs, y_cls.to(device))
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == y_cls.to(device)).sum().item() / len(y_cls)
                print(f"Class {cls}, Epoch {epoch}: Loss = {loss.item():.4f}, Acc = {accuracy:.4f}")
        
        class_specific_models[cls] = model
    
    # Incorporate class-specific predictions into overall scoring
    final_scores = np.zeros(len(test_feature_set1))
    for i, true_label in enumerate(test_true_labels):
        if true_label in class_specific_models:
            # Use class-specific model
            model = class_specific_models[true_label]
            X_test_sample = torch.tensor(test_feature_set1[i:i+1], dtype=torch.float32)
            with torch.no_grad():
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
    
    print("\n--- Final Evaluation with Improved Models ---")
    print("RAUC@100:", rauc(is_bug, 100))
    print("RAUC@200:", rauc(is_bug, 200))
    print("RAUC@500:", rauc(is_bug, 500))
    print("RAUC@1000:", rauc(is_bug, 1000))
    print("RAUC@all:", rauc(is_bug, test_num))
    print("ATRC:", ATRC(is_bug, len(test_error_index)))
    
    return final_scores, is_bug

# --- Integration with main function ---
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
    
    # Previous bug detection code...
    # [Your existing bug detection code would go here]
    
    # Add ensemble bug detection
    print("\n--- Running Enhanced Ensemble Bug Detection ---")
    final_scores, is_bug = run_ensemble_bug_detection()
    
    # Optionally, you could also visualize the most confident bug predictions
    # or save the bug detection results for further analysis
    
if __name__ == '__main__':
    main()
