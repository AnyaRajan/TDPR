import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from omegaconf import OmegaConf

# Import your data utilities; ensure these functions are defined in data_util.py
from data_util import (
    get_train_data,
    get_val_and_test,
    calculate_avg_pro_diff,
    calculate_avg_info,
    calculate_std_info,
    calculate_label_std,
    get_num_of_most_diff_class
)

# Dummy definitions for rauc and ATRC (replace with your own implementations)
def rauc(is_bug_ranked, K):
    """
    Dummy RAUC metric. Replace with your actual implementation.
    is_bug_ranked: 1D numpy array of binary labels ordered by predicted ranking.
    K: cutoff index.
    """
    # For demonstration, return a random value.
    return np.random.rand()

def ATRC(is_bug_ranked, num_errors):
    """
    Dummy ATRC metric. Replace with your actual implementation.
    Returns a tuple: (overall_metric, list_of_cutoff_metrics)
    """
    overall = np.random.rand()
    cutoff_list = [np.random.rand() for _ in range(10)]
    return overall, cutoff_list

# Load configuration from config.yaml; ensure this file contains keys like "model" and "corruption"
conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_features(pros, labels, infos):
    """
    Extract features from model predictions and uncertainty information.
    Assumes:
      pros: numpy array of shape [num_samples, num_aug, num_classes] 
            or [num_samples, num_classes] (in which case it will be expanded to have one augmentation).
      labels: numpy array of predicted labels.
      infos: numpy array of uncertainty values (one per sample).
    Returns a normalized feature matrix.
    """
    # If pros is 2D (i.e. only one augmentation per sample), add an extra dimension.
    if pros.ndim == 2:
        pros = np.expand_dims(pros, axis=1)  # Now shape is [num_samples, 1, num_classes]
    
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = calculate_avg_info(infos)
    std_info = calculate_std_info(infos)
    std_label = calculate_label_std(labels)
    max_diff_num = get_num_of_most_diff_class(labels)
    
    feature = np.column_stack((std_label, avg_info, std_info, max_diff_num, avg_p_diff))
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    return feature


# Define the Ranking Neural Network (a simple 2-layer MLP)
class RankingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RankingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Output a single continuous ranking score
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        score = self.fc2(x)
        return score


def train_ranking_model(model, X_train, y_train, epochs=50, margin=1.0):
    """
    Train the ranking network using MarginRankingLoss.
    
    X_train: torch.Tensor of shape [N, feature_dim]
    y_train: torch.Tensor of shape [N] with binary labels (1 for bug, 0 for non-bug)
    margin: the desired margin between positive and negative sample scores.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    margin_loss = nn.MarginRankingLoss(margin=margin)

    for epoch in range(epochs):
        optimizer.zero_grad()
        scores = model(X_train).squeeze()  # Shape: [N]

        # Find indices of positive (bug) and negative samples.
        pos_indices = (y_train == 1).nonzero(as_tuple=True)[0]
        neg_indices = (y_train == 0).nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            print("Not enough positive or negative samples for ranking.")
            continue

        # Get scores for each group.
        pos_scores = scores[pos_indices]
        neg_scores = scores[neg_indices]

        # Create all pairs: for every positive sample, pair with every negative sample.
        pos_scores_exp = pos_scores.unsqueeze(1).expand(-1, len(neg_scores))
        neg_scores_exp = neg_scores.unsqueeze(0).expand(len(pos_scores), -1)

        # The target tensor: we want pos_score - neg_score to be at least margin.
        target = torch.ones_like(pos_scores_exp)

        loss = margin_loss(pos_scores_exp, neg_scores_exp, target)
        loss.backward()
        optimizer.step()

        print(f"Ranking Model - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def main():
    # ---------------------------
    # Data Loading and Feature Extraction
    # ---------------------------
    # Load validation and test data loaders. get_val_and_test should return (valloader, testloader).
    valloader, testloader = get_val_and_test(conf.corruption)
    
    # Define a test function to get classifier outputs.
    def test(net, loader):
        net.eval()
        pros = []
        labels = []
        infos = []
        error_index = []
        sample_idx = 0  # To track global sample indices
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                pro = F.softmax(outputs, dim=1).cpu().numpy()
                pros.extend(pro)
                # Compute information entropy for each sample
                info = -np.sum(pro * np.log2(pro + 1e-12), axis=1)
                infos.extend(info)
                _, predicted = outputs.max(1)
                labels.extend(predicted.cpu().numpy())
                
                # Record error indices (if prediction does not match target)
                incorrect_mask = ~predicted.eq(targets)
                if incorrect_mask.any():
                    # Assuming loader has a batch_size attribute; otherwise, use len(inputs)
                    batch_size = loader.batch_size if hasattr(loader, "batch_size") else inputs.size(0)
                    incorrect_indices = sample_idx + torch.nonzero(incorrect_mask).view(-1)
                    error_index.extend(incorrect_indices.cpu().tolist())
                sample_idx += inputs.size(0)
        return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

    # Load your classifier model from the models module.
    import models  # Ensure that models is in your PYTHONPATH and conf.model is valid.
    net = models.__dict__[conf.model]().to(device)
    
    # Optionally, load pretrained weights if available:
    # net.load_state_dict(torch.load("path/to/pretrained_weights.pth"))
    
    # Get classifier outputs for validation and test sets.
    val_pros, val_labels, val_infos, val_error_index = test(net, valloader)
    test_pros, test_labels, test_infos, test_error_index = test(net, testloader)
    
    # Extract features from classifier outputs.
    val_features = extract_features(val_pros, val_labels, val_infos)
    test_features = extract_features(test_pros, test_labels, test_infos)
    
    # Create binary labels for ranking: 1 for bug-revealing samples, 0 for non-bug.
    val_bug_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_bug_labels[val_error_index] = 1

    # ---------------------------
    # Ranking Network Training
    # ---------------------------
    # Convert features and labels to torch tensors.
    X_train = torch.tensor(val_features, dtype=torch.float32)
    y_train = torch.tensor(val_bug_labels, dtype=torch.float32)  # using float for MarginRankingLoss

    # Define the ranking network.
    input_dim = X_train.shape[1]
    ranking_model = RankingNet(input_dim=input_dim, hidden_dim=64).to(device)

    # Train the ranking model.
    train_ranking_model(ranking_model, X_train.to(device), y_train.to(device), epochs=50, margin=1.0)

    # ---------------------------
    # Inference and Ranking Evaluation
    # ---------------------------
    ranking_model.eval()
    with torch.no_grad():
        X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
        # Obtain continuous ranking scores.
        scores = ranking_model(X_test).squeeze().cpu().numpy()

    # Sort test samples by their scores in descending order.
    test_num = len(testloader.dataset)
    # Create binary labels for the test set: 1 if bug-revealing, 0 otherwise.
    is_bug = np.zeros(test_num, dtype=int)
    is_bug[test_error_index] = 1

    # Get ranking order based on scores.
    sorted_indices = np.argsort(scores)[::-1]
    is_bug_ranked = is_bug[sorted_indices]

    # Evaluate ranking with your metrics.
    print("RAUC@100:", rauc(is_bug_ranked, 100))
    print("RAUC@200:", rauc(is_bug_ranked, 200))
    print("RAUC@500:", rauc(is_bug_ranked, 500))
    print("RAUC@1000:", rauc(is_bug_ranked, 1000))
    print("RAUC@all:", rauc(is_bug_ranked, test_num))
    print("ATRC:", ATRC(is_bug_ranked, len(test_error_index)))


if __name__ == '__main__':
    main()
