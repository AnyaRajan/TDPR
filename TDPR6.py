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

# Import your data utilities and metrics functions (e.g., rauc, ATRC)
from data_util import (
    get_train_data,
    get_val_and_test,
    calculate_avg_pro_diff,
    calculate_avg_info,
    calculate_std_info,
    calculate_label_std,
    get_num_of_most_diff_class
)

# Load configuration
conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_features(pros, labels, infos):
    """
    Extract features from model predictions and other statistics.
    Adjust this function according to your needs.
    """
    # Assuming pros is a numpy array with shape [num_samples, num_aug, num_classes]
    # If necessary, transpose to match your expectation.
    # For example, here we assume the input is already in proper shape.
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = calculate_avg_info(infos)
    std_info = calculate_std_info(infos)
    std_label = calculate_label_std(labels)
    max_diff_num = get_num_of_most_diff_class(labels)
    feature = np.column_stack((std_label, avg_info, std_info, max_diff_num, avg_p_diff))
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    return feature


# Define the Ranking Neural Network
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


def train_ranking_model(model, X_train, y_train, epochs=15, margin=1.0):
    """
    Train the ranking network using MarginRankingLoss.
    
    Parameters:
      X_train: torch.Tensor of shape [N, feature_dim]
      y_train: torch.Tensor of shape [N] with binary labels (1 for bug, 0 for non-bug)
      margin: the minimum desired difference between bug and non-bug scores.
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

        # Create all pairs: positive should be higher than negative.
        pos_scores_exp = pos_scores.unsqueeze(1).expand(-1, len(neg_scores))
        neg_scores_exp = neg_scores.unsqueeze(0).expand(len(pos_scores), -1)

        # The target for MarginRankingLoss is 1: pos_score should exceed neg_score.
        target = torch.ones_like(pos_scores_exp)

        loss = margin_loss(pos_scores_exp, neg_scores_exp, target)
        loss.backward()
        optimizer.step()

        print(f"Ranking Model - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


def main():
    # ---------------------------
    # Data Loading and Feature Extraction
    # ---------------------------
    # Load training data for the base classifier if needed to extract features.
    # Here we assume you already trained a classifier and extracted its outputs.
    # For demonstration, we use the validation set to extract features.
    valloader, testloader = get_val_and_test(conf.corruption)
    
    # Here, we assume your test() function runs the classifier to get probabilities,
    # predicted labels, and some measure of uncertainty (infos).
    def test(net, loader):
        net.eval()
        pros = []
        labels = []
        infos = []
        error_index = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                pro = F.softmax(outputs, dim=1).cpu().numpy()
                pros.extend(pro)
                # Calculate information entropy for each sample
                info = -np.sum(pro * np.log2(pro + 1e-12), axis=1)
                infos.extend(info)
                _, predicted = outputs.max(1)
                labels.extend(predicted.cpu().numpy())
                # Record error indices (if prediction does not match target)
                incorrect_mask = ~predicted.eq(targets)
                if incorrect_mask.any():
                    incorrect_indices = (batch_idx * loader.batch_size) + torch.nonzero(incorrect_mask).view(-1)
                    error_index.extend(incorrect_indices.tolist())
        return np.array(pros), np.array(labels), np.array(infos), np.array(error_index)

    # For this demo, we assume the classifier is already trained and saved.
    # Load your classifier model (this may be a CNN, for example) from your models module.
    # Replace `conf.model` with your classifier model name.
    import models  # your classifier is defined here
    net = models.__dict__[conf.model]().to(device)
    
    # Optionally load pretrained weights for net if available.
    # net.load_state_dict(torch.load("path/to/pretrained_weights.pth"))
    
    # Get classifier outputs from validation and test sets.
    val_pros, val_labels, val_infos, val_error_index = test(net, valloader)
    test_pros, test_labels, test_infos, test_error_index = test(net, testloader)
    
    # Extract features from the classifier outputs
    val_features = extract_features(val_pros, val_labels, val_infos)
    test_features = extract_features(test_pros, test_labels, test_infos)
    
    # Create binary labels for ranking: 1 for bug-revealing samples, 0 otherwise.
    # Here, we assume error indices indicate bug-revealing samples.
    val_bug_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_bug_labels[val_error_index] = 1

    # ---------------------------
    # Ranking Network Training
    # ---------------------------
    # Convert features and labels to torch tensors.
    X_train = torch.tensor(val_features, dtype=torch.float32)
    y_train = torch.tensor(val_bug_labels, dtype=torch.float32)  # using float for ranking loss

    # Define the ranking network.
    input_dim = X_train.shape[1]
    ranking_model = RankingNet(input_dim=input_dim, hidden_dim=64).to(device)

    # Train the ranking model using pairwise margin ranking loss.
    train_ranking_model(ranking_model, X_train.to(device), y_train.to(device), epochs=50, margin=1.0)

    # ---------------------------
    # Inference and Ranking Evaluation
    # ---------------------------
    ranking_model.eval()
    with torch.no_grad():
        X_test = torch.tensor(test_features, dtype=torch.float32).to(device)
        # Obtain continuous ranking scores
        scores = ranking_model(X_test).squeeze().cpu().numpy()

    # Sort test samples by their scores in descending order.
    test_num = len(testloader.dataset)
    # Create binary labels for test set using error indices (1 if bug-revealing, 0 otherwise)
    is_bug = np.zeros(test_num, dtype=int)
    is_bug[test_error_index] = 1

    # Get ranking order based on scores
    sorted_indices = np.argsort(scores)[::-1]
    is_bug_ranked = is_bug[sorted_indices]

    # Evaluate ranking with your metrics, e.g., RAUC and ATRC.
    # These functions should be defined in your codebase.
    print("RAUC@100:", rauc(is_bug_ranked, 100))
    print("RAUC@200:", rauc(is_bug_ranked, 200))
    print("RAUC@500:", rauc(is_bug_ranked, 500))
    print("RAUC@1000:", rauc(is_bug_ranked, 1000))
    print("RAUC@all:", rauc(is_bug_ranked, test_num))
    print("ATRC:", ATRC(is_bug_ranked, len(test_error_index)))


if __name__ == '__main__':
    main()
