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
from BugNet import *
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


conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_trc_curve(sorted_flags):
    trc_values = []
    bug_count = int(np.sum(sorted_flags))
    for i in range(1, len(sorted_flags) + 1):
        selected = sorted_flags[:i]
        trc = np.sum(selected) / min(i, bug_count) if bug_count > 0 else 0
        trc_values.append(trc)
    return trc_values

def plot_trc_curve(sorted_flags_list, labels, title="Bug Discovery Curve (TRC)", figsize=(8, 6)):
    plt.figure(figsize=figsize)
    for sorted_flags, label in zip(sorted_flags_list, labels):
        trc_curve = compute_trc_curve(sorted_flags)
        plt.plot(trc_curve, label=label)
    plt.xlabel("Number of Samples Selected")
    plt.ylabel("TRC (Bug Recall)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def update_val_labels_with_unstable(val_labels, val_label_arrays, threshold=0.5):
    """
    Marks unstable samples (based on std deviation of predicted labels across augmentations)
    as bug-prone (label = 1).
    
    Parameters:
        val_labels (np.ndarray): Current binary labels for validation samples (0 or 1).
        val_label_arrays (np.ndarray): Shape (N, num_augs), predicted labels across augmentations.
        threshold (float): Standard deviation threshold to consider a sample unstable.

    Returns:
        np.ndarray: Updated labels with unstable samples marked as bug-prone.
    """
    std_dev = np.std(val_label_arrays, axis=1)
    unstable_flags = (std_dev > threshold).astype(int)
    
    updated_labels = val_labels.copy()
    updated_labels[unstable_flags == 1] = 1
    return updated_labels


def forward_with_augmentations(net, sample, num_aug=conf.augs):
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

def generate_augmented_outputs(net, dataset, num_aug=conf.augs):
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

def extract_features(pros, labels, infos):
    # --- Existing features ---
    avg_p_diff = calculate_avg_pro_diff(pros)                       # Cosine similarity distance
    avg_info = np.mean(infos, axis=1)                               # Mean entropy
    std_info = np.std(infos, axis=1)                                # Entropy variation
    std_label = np.std(labels, axis=1)                              # Label variance
    max_diff_num = get_num_of_most_diff_class(labels)              # Max disagreement
    # --- New features ---
    kl_divs = calculate_kl_divergence(pros)                         # KL divergence to last
    agreements = calculate_agreement(labels)                        # Mode agreement
    margins = calculate_margin(pros)                                # Top-2 prediction margin
    mi_scores = calculate_mutual_information(pros)      

    # --- Combine features ---
    feature = np.column_stack((
        std_label,
        avg_info,
        std_info,
        max_diff_num,
        avg_p_diff,
        kl_divs,
        agreements,
        margins,
        mi_scores,
    ))

    # Normalize
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
    print(f"\n🧪 Final Test Accuracy: {acc:.2f}%")
    return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

def train(net, num_epochs, optimizer, criterion, trainloader, device):
    net.to(device)
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
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
        scheduler.step()  # Step the scheduler after each epoch

        print(f"✅ Epoch {epoch + 1}: Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        


# --- Main Function ---
def main():
    # Initialize model and training data.
    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    if conf.dataset in ["cifar10", "imagenet"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), weight_decay=1e-4, momentum=0.9, lr=0.1)
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
        val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=conf.augs)
        test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=conf.augs)
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

    # New: label unstable samples as bug-prone too
    val_labels = update_val_labels_with_unstable(val_labels, val_label_arrays, threshold=0.5)

    # Hyperparameter for hidden layer size
    hidden_dim = 64  # You can make this a configurable argument

    # Convert data to PyTorch tensors
    X_train = torch.tensor(val_features, dtype=torch.float32)
    y_train = torch.tensor(val_labels, dtype=torch.long)
    X_test = torch.tensor(test_features, dtype=torch.float32)

    # Choose model: "rf" or "nn"
    model_type = conf.model_type    

        # Binary ground-truth bug flags
    test_flags = np.zeros(len(testloader.dataset))
    test_flags[test_error_index] = 1

    # --- Train BugNet (NN) ---
    bugnet = BugNet(input_dim=val_features.shape[1], hidden_dim=128).to(device)
    # Convert labels to float for BCEWithLogitsLoss
    y_train_float = y_train.float()

    # Compute positive class weight
    pos_weight = torch.tensor([(len(y_train_float) - y_train_float.sum()) / y_train_float.sum()]).to(device)

    # Define loss for binary classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(bugnet.parameters(), lr=0.001)

    for epoch in range(50):
        bugnet.train()
        optimizer.zero_grad()
        outputs = bugnet(X_train.to(device))
        loss = criterion(outputs, y_train_float.to(device))
        loss.backward()
        optimizer.step()
        #calculate accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        correct = (predicted.cpu() == y_train).sum().item()
        total = y_train.size(0)
        acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")
        
    bugnet.eval()
    with torch.no_grad():
        # probs_bugnet = F.softmax(bugnet(X_test.to(device)), dim=1)
        # scores_bugnet = probs_bugnet[:, 1].cpu().numpy()
        logits = bugnet(X_test.to(device))
        probs_bugnet = torch.sigmoid(logits).cpu().numpy()
        scores_bugnet = probs_bugnet  # since sigmoid output is the score

        

    # --- Train RandomForest ---
    # Use best hyperparameters from grid search
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight='balanced')
    rf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    scores_rf = rf.predict_proba(X_test.cpu().numpy())[:, 1]
    
    # --- Train XGBoost ---
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        # scale_pos_weight=((len(y_train_float) - y_train_float.sum()) / y_train_float.sum()).item(),
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    scores_xgb = xgb.predict_proba(X_test.cpu().numpy())[:, 1]



    # --- Ensemble ---
    scores_ensemble = ensemble_scores(scores_bugnet, scores_rf, scores_xgb)

    # --- Evaluate All Models ---
    model_scores = {
        "BugNet": scores_bugnet,
        "RandomForest": scores_rf,
        "XGBoost": scores_xgb,
        "Ensemble": scores_ensemble
    }

    summary = evaluate_models(model_scores, test_flags)

    print("\n📊 Evaluation Summary (Auto-selected best based on composite score):")
    for entry in summary:
        print(f"{entry['name']}: RAUC@100={entry['RAUC@100']:.3f}, RAUC@200={entry['RAUC@200']:.3f}, RAUC@500={entry['RAUC@500']:.3f}, RAUC@1000={entry['RAUC@1000']:.3f}, RAUC@all={entry['RAUC@all']:.3f}, "
              f"ATRC={entry['ATRC']:.3f}, Composite Score={entry['Composite Score']:.3f}")

    # --- Plot Bug Discovery Curve (TRC) ---
    sorted_flags_list = [entry["Sorted Flags"] for entry in summary]
    labels = [entry["name"] for entry in summary]
    plot_trc_curve(sorted_flags_list, labels)

if __name__ == '__main__':
    main()
