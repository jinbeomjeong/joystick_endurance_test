import numpy as np
from tqdm.auto import tqdm


def create_seq_dataset_multiple_input_single_output(data: np.ndarray, seq_len=1, pred_distance=0, target_idx_pos=1):
    feature, target = [], []

    for i in tqdm(range(data.shape[0] - pred_distance), desc='creating sequence dataset...'):
        if i+1 >= seq_len:
            feature.append(data[i+1-seq_len:i+1, 0:target_idx_pos])

            if target_idx_pos >= 0:
                target.append(data[i + pred_distance, target_idx_pos:])

    return np.array(feature), np.array(target)  # data shape(n_samples, seq_len, n_features), seq len=[t-29, t-28, t-27,..., t0]