import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_util import get_augmentation_pipeline

def calculate_info_entropy_from_probs(probs):
    return -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)

def forward_with_augmentations(net, sample, num_aug=100):
    if isinstance(sample, torch.Tensor):
        sample = transforms.ToPILImage()(sample.cpu())
    
    aug_pipeline = get_augmentation_pipeline()
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
