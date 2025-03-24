import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def calculate_avg_pro_diff(pros):
    num_samples, num_aug, _ = pros.shape
    avg_diffs = np.zeros(num_samples)
    for i in range(num_samples):
        ref = pros[i, -1, :].reshape(1, -1)
        sims = cosine_similarity(pros[i, :num_aug-1, :], ref)
        distances = 1 - sims.flatten()
        avg_diffs[i] = np.mean(distances)
    return avg_diffs

def get_num_of_most_diff_class(labels):
    num_samples, num_aug = labels.shape
    max_diff = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        target = labels[i, -1]
        diff_counts = {}
        for j in range(num_aug - 1):
            if labels[i, j] != target:
                diff_counts[labels[i, j]] = diff_counts.get(labels[i, j], 0) + 1
        max_diff[i] = max(diff_counts.values()) if diff_counts else 0
    return max_diff

def extract_features(pros, labels, infos):
    print("Extracting features...")  # Debugging Print

    avg_p_diff = calculate_avg_pro_diff(pros)
    avg_info = np.mean(infos, axis=1)
    std_info = np.std(infos, axis=1)
    std_label = np.std(labels, axis=1)
    max_diff_num = get_num_of_most_diff_class(labels)

    feature = np.column_stack((std_label, avg_info, std_info, max_diff_num, avg_p_diff))

    print(f"Feature Shape: {feature.shape}")  # Debugging Print
    print(f"Sample Features (First 5 Rows):\n{feature[:5]}")  # Print first few features

    scaler = MinMaxScaler()
    return scaler.fit_transform(feature)

