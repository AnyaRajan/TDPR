import numpy as np
import xgboost
from xgboost import DMatrix
import torch.optim as optim
import torch.nn as nn

from model_utils import initialize_model, get_data_loaders
from generate_outputs import generate_augmented_outputs
from feature_extraction import extract_features
from train import train
from test import test

def main():
    net = initialize_model()
    trainloader, valloader, testloader = get_data_loaders()
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training the model
    train(net, optimizer, criterion, trainloader)

    # Generating outputs for validation dataset
    val_prob_arrays, val_label_arrays, val_uncertainty_arrays = generate_augmented_outputs(net, valloader.dataset, num_aug=50)
    val_features = extract_features(val_prob_arrays, val_label_arrays, val_uncertainty_arrays)
    
    # Evaluating the model
    _, _, _, val_error_index = test(net, valloader)
    
    val_labels = np.zeros(len(valloader.dataset), dtype=int)
    val_labels[val_error_index] = 1

    # Train XGBoost
    print("Training XGBoost ranking model...")
    train_data = DMatrix(val_features, label=val_labels)
    xgb_rank_params = {'objective': 'rank:pairwise', 'max_depth': 5, 'learning_rate': 0.05}
    rankModel = xgboost.train(xgb_rank_params, train_data)

    # Predict scores on test dataset
    print("Predicting scores on test data...")
    test_prob_arrays, test_label_arrays, test_uncertainty_arrays = generate_augmented_outputs(net, testloader.dataset, num_aug=50)
    test_features = extract_features(test_prob_arrays, test_label_arrays, test_uncertainty_arrays)
    test_data = DMatrix(test_features)
    scores = rankModel.predict(test_data)
    print(f"Sample Scores (First 10): {scores[:10]}")  # Debugging Print

    print("Model training complete.")

if __name__ == '__main__':
    main()
