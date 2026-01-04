import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def parse_pickle_labels(file_path):
    """Parse pickle format label file and return dictionary"""
    with open(file_path, 'rb') as f:
        label_dict = pickle.load(f)
    return label_dict


def des_videos(txt_path):
    """Get list of video IDs to be calculated from text file"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    video_list = [line.strip() for line in lines if line.strip()]
    return set(video_list)


def calculate_ts_align(predict_list, true_list):
    """
    Calculate TS_align metric for a single video

    Args:
        predict_list (list): Predicted labels list (0=normal, 1=abnormal)
        true_list (list): True labels list (0=normal, 1=abnormal)

    Returns:
        float: TS_align value
    """
    N = len(predict_list)
    # Check length consistency
    if len(true_list) != N:
        # Use minimum length
        min_len = min(N, len(true_list))
        predict_list = predict_list[:min_len]
        true_list = true_list[:min_len]
        N = min_len

    # Ensure at least 2 frames to calculate
    if N < 2:
        return None

    # Compute e_i = l_i - t_i
    e = [p - t for p, t in zip(predict_list, true_list)]

    # Compute Δe_i = |e_i - e_{i-1}|
    delta_e = []
    for i in range(1, N):
        delta = abs(e[i] - e[i - 1])
        delta_e.append(delta)

    # Compute TS_align = 1 - (1/(N-1)) * ∑|Δe_i|
    sum_delta = sum(delta_e)
    ts_align = 1 - (1 / (N - 1)) * sum_delta

    return ts_align


def calculate_f1(video_pred_dict, video_true_dict, selected_videos):
    """
    Calculate F1 score and other metrics for all videos

    Args:
        video_pred_dict (dict): Video-level prediction labels dictionary {videoID: label(0 or 1)}
        video_true_dict (dict): Video-level true labels dictionary {videoID: label(0 or 1)}
        selected_videos (set): Set of video IDs to be calculated

    Returns:
        dict: Dictionary containing multiple metrics
    """
    # Initialize confusion matrix (based on custom definition)
    TP = 0  # Predicted 0 and true 0 (0 as positive class)
    FP = 0  # Predicted 0 but true 1
    FN = 0  # Predicted 1 but true 0
    TN = 0  # Predicted 1 and true 1

    missing_count = 0
    common_ids = set()
    # Create temporary list for filtering results
    filtered_videos = []

    for vid in selected_videos:
        # Check if video ID exists in both dictionaries
        if vid not in video_pred_dict or vid not in video_true_dict:
            missing_count += 1
            continue

        # Get true value
        true_val = int(video_true_dict[vid][0])
        
        # Only process videos with true_val = 0
        if true_val == 0:
            # Get predicted value
            pred_val = int(video_pred_dict[vid][0])
            
            # Add qualified videos to filtered list
            filtered_videos.append(vid)
            
    for vid in filtered_videos:

        pred_val = int(video_pred_dict[vid][0])
        true_val = int(video_true_dict[vid][0])

        # Calculate F1 for negative class
        if true_val == 0:
            if pred_val == 0:
                TP += 1  # Correct positive prediction
            else:  # pred_val == 1
                FN += 1  # Missed positive class (false negative)
        else:  # true_val == 1
            if pred_val == 0:
                FP += 1  # False positive
            else:  # pred_val == 1
                TN += 1  # Correct negative prediction
        
        common_ids.add(vid)

    if not common_ids:
        print("Warning: No common video IDs found in predicted and true labels")
        return None

    # Calculate metrics and print results (including confusion matrix)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


    return f1