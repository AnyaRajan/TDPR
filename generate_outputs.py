import numpy as np
from augmentation import forward_with_augmentations

def generate_augmented_outputs(net, dataset, num_aug=50):
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
