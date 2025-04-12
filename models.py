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

def run_rf_grid(X_train, y_train, X_test, test_error_index):
    test_flags = np.zeros(len(X_test))
    test_flags[test_error_index] = 1

    results = []
    grid = ParameterGrid({
        'n_estimators': [50, 100],
        'max_depth': [4, 8, None]
    })

    for params in grid:
        model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                       max_depth=params['max_depth'],
                                       class_weight='balanced')
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        ranking = np.argsort(scores)[::-1]
        sorted_flags = test_flags[ranking]

        rauc_100 = rauc(sorted_flags, 100)
        rauc_200 = rauc(sorted_flags, 200)
        rauc_500 = rauc(sorted_flags, 500)
        rauc_1000 = rauc(sorted_flags, 1000)
        rauc_all = rauc(sorted_flags, len(test_flags))
        atrc_val, _ = ATRC(sorted_flags, int(np.sum(test_flags)))
        results.append((params, rauc_100, rauc_200, rauc_500, rauc_1000, rauc_all, atrc_val))

    return results

class BugNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(BugNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.out = nn.Linear(hidden_dim // 2, 1)  # Binary classification → single logit

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.out(x).squeeze(1)  # Output shape: (batch,)
