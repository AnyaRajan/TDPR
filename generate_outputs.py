def generate_augmented_outputs(net, dataset, num_aug=50):
    print("Generating augmented outputs...")  # Debugging Print
    num_samples = len(dataset)
    
    first_sample, _ = dataset[0]
    if isinstance(first_sample, torch.Tensor):
        first_sample = first_sample.to(device)
    with torch.no_grad():
        num_classes = net(first_sample.unsqueeze(0)).shape[1]
    
    print(f"Dataset Size: {num_samples}, Number of Classes: {num_classes}")  # Debugging Print

    all_prob_arrays = np.zeros((num_samples, num_aug, num_classes))

    for idx in range(num_samples):
        sample, _ = dataset[idx]
        probs, _, _ = forward_with_augmentations(net, sample, num_aug=num_aug)
        all_prob_arrays[idx] = probs

        if idx % 100 == 0:  # Print every 100 samples
            print(f"Sample {idx}/{num_samples} Processed.")  

    return all_prob_arrays
