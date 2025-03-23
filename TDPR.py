from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
from sklearn.preprocessing import MinMaxScaler
from xgboost import DMatrix
import xgboost
from data_util import *
from omegaconf import OmegaConf
import models

conf = OmegaConf.load('config.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from PIL import Image

# --- Augmentation and Forward Pass Functions ---

def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def forward_with_augmentations(net, sample, num_aug=50):
    # Assume sample is a tensor; convert to PIL image if needed.
    if isinstance(sample, torch.Tensor):
        sample = transforms.ToPILImage()(sample.cpu())
    
    aug_pipeline = get_augmentation_pipeline()  # Defined in data_util.py
    prob_list = []
    label_list = []
    uncertainty_list = []
    
    net.eval()
    with torch.no_grad():
        for _ in range(num_aug):
            aug_sample = aug_pipeline(sample)
            aug_sample = aug_sample.unsqueeze(0).to(device)
            outputs = net(aug_sample)
            probs = F.softmax(outputs, dim=1).cpu().numpy().squeeze(0)
            prob_list.append(probs)
            label_list.append(np.argmax(probs))
            uncertainty_list.append(calculate_info_entropy_from_probs(probs))
            
    return np.array(prob_list), np.array(label_list), np.array(uncertainty_list)

def generate_augmented_outputs(net, dataset, num_aug=50):
    all_prob_arrays = []
    all_label_arrays = []
    all_uncertainty_arrays = []
    for idx in range(len(dataset)):
        sample, _ = dataset[idx]  # Use label if needed; here we only use the sample.
        probs, labels, uncertainties = forward_with_augmentations(net, sample, num_aug=num_aug)
        all_prob_arrays.append(probs)
        all_label_arrays.append(labels)
        all_uncertainty_arrays.append(uncertainties)
    return np.array(all_prob_arrays), np.array(all_label_arrays), np.array(all_uncertainty_arrays)

# --- Feature Extraction (Modified to work with augmented outputs) ---
def extract_features(pros, labels, infos):
    # Here, 'pros' is an array of shape (num_samples, num_aug, num_classes)
    # 'labels' is (num_samples, num_aug) and 'infos' is (num_samples, num_aug)
    pros = pros.transpose([1, 0, 2])  # Transpose similar to original expectations.
    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = calculate_avg_info(infos)
    std_info = calculate_std_info(infos)
    std_label = calculate_label_std(labels)
    max_diff_num = get_num_of_most_diff_class(labels)
    feature = np.column_stack((
        std_label,
        avg_info,
        std_info,
        max_diff_num,
        avg_p_diff
    ))
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    return feature

def calculate_info_entropy(pros):
    entropys = []
    for pro in pros:
        entropy = -np.sum(pro * np.log2(pro))
        entropys.append(entropy)
    return entropys

# --- Original Test Function (unchanged) ---
def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    pros = []
    labels = []
    infos = []
    error_index = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            pro = F.softmax(outputs, dim=1).cpu().numpy()
            pros.extend(pro)
            info = calculate_info_entropy(pro)
            infos.extend(info)
            _, predicted = outputs.max(1)
            labels.extend(predicted.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            incorrect_mask = ~predicted.eq(targets)
            if incorrect_mask.any():
                incorrect_indices = (batch_idx * testloader.batch_size) + torch.nonzero(incorrect_mask).view(-1)
                error_index.extend(incorrect_indices.tolist())
        acc = 100. * correct / total
    return pros, labels, infos, error_index

# --- Training Function ---
def train(net, num_epochs, optimizer, criterion, trainloader):
    snapshot_root = Path("snapshots") / Path(str(conf.model))
    snapshot_root.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        net.train()
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Commenting out snapshot saving for augmented evaluation
        # checkpoint_path = snapshot_root / f'epoch_{epoch}.pth'
        # torch.save(net.state_dict(), checkpoint_path)
        # print(f"Checkpoint saved: {checkpoint_path}")

# --- Main Function ---
def main():
    # Create snapshot directory (still used if needed)
    snapshot_root = Path("snapshots") / Path(conf.model)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    net = models.__dict__[conf.model]().to(device)
    trainloader = get_train_data(conf.dataset)
    
    # Set up criterion and optimizer based on dataset
    if conf.dataset in ["cifar10", "imagenet"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), weight_decay=5e-4, momentum=0.9, lr=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train the model (snapshot saving is commented out)
    train(net, conf.epochs, optimizer, criterion, trainloader)

    # Optionally: Load final checkpoint if you saved one; here we use the final trained model.
    # For example, you could load the last checkpoint if desired.
    # epoch_path = snapshot_root / f'epoch_{conf.epochs - 1}.pth'
    # net.load_state_dict(torch.load(epoch_path))

    # Generate augmented outputs for each sample in the test dataset.
    # Here we assume get_clean_test_dataset returns a DataLoader; we convert it to a Dataset.
    testloader = get_clean_test_dataset(conf.dataset)
    test_dataset = testloader.dataset  # Retrieve the underlying dataset

    # Generate outputs using 50 augmentations per sample.
    prob_arrays, label_arrays, uncertainty_arrays = generate_augmented_outputs(net, test_dataset, num_aug=50)
    
    # Extract features based on the augmented outputs.
    features = extract_features(prob_arrays, label_arrays, uncertainty_arrays)
    print("Extracted features shape:", features.shape)
    
    # Continue with ranking model building using XGBoost etc.
    # For example:
    # dmatrix = DMatrix(features, label=your_labels_here)
    # model = xgboost.train(params, dmatrix, num_boost_round=...)
    # ... etc.
    
if __name__=='__main__':
    main()
