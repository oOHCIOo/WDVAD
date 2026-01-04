import pickle
import os
from test import test, scorebinary
import numpy as np
import torch
from utils import anomap
import copy
from MMD import permutation_test_mmd
from eval import des_videos, calculate_ts_align, calculate_f1
import pandas as pd
import json

def load_pickle(file_path):
    """Load data from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_MMD_and_update_labels(args, train_loader, model, original_model, device, 
                                    video_labels_path, frame_labels_path, 
                                    current_round=1, dataset_name='dataset'):
    """
    Calculate MMD distance and update labels based on statistical significance
    
    Args:
        args: Command line arguments
        train_loader: DataLoader for training data
        model: Current model after training
        original_model: Original model before training
        device: Computation device (CPU/GPU)
        video_labels_path: Path to video-level labels
        frame_labels_path: Path to frame-level labels
        current_round: Current FL round
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of paths to new video and frame label files
    """
    # Normalize dataset_type to remove trailing underscore if present
    dataset_type = args.dataset_type.rstrip('_') if hasattr(args, 'dataset_type') else 'shanghaitech'
    
    # Verify models are distinct
    if model is original_model:
        print("Warning: Current model and original model are the same instance!")
    print(f"Model ID: {id(model)}, Original Model ID: {id(original_model)}")
    print(f"Weight difference: {torch.sum(model.fc.weight != original_model.fc.weight).item()}")
    
    # Load original labels
    video_labels = load_pickle(video_labels_path)
    frame_labels = load_pickle(frame_labels_path)

    def calculate_MMD(original_predict_dict, new_predict_dict):
        """
        Compute MMD (Maximum Mean Discrepancy) between original and new predictions
        
        Args:
            original_predict_dict: Predictions from original model
            new_predict_dict: Predictions from current model
            
        Returns:
            Dictionary of MMD p-values for each video
        """
        score_MMD = {}
        common_video_ids = set(new_predict_dict.keys()) & set(original_predict_dict.keys())
        print(f"Calculating MMD for {len(common_video_ids)} common videos")

        for video_id in common_video_ids:
            # Prepare data for MMD calculation
            X = original_predict_dict[video_id].reshape(-1, 1)
            Y = new_predict_dict[video_id].reshape(-1, 1)
            
            # Compute MMD and p-value
            _, p_value = permutation_test_mmd(X, Y, sigma=1.0, num_permutations=1000)
            score_MMD[video_id] = p_value

        return score_MMD

    # Create output directory for new labels
    output_dir = os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', 'new_label')
    os.makedirs(output_dir, exist_ok=True)
    new_video_filename = os.path.join(output_dir, 'new_video_label.pickle')
    new_frame_filename = os.path.join(output_dir, 'new_frame_label.pickle')
    
    # Return existing files if already generated
    if os.path.exists(new_video_filename) and os.path.exists(new_frame_filename):
        print(f"Label files already exist, returning:\nVideo labels: {new_video_filename}\nFrame labels: {new_frame_filename}")
        return new_video_filename, new_frame_filename

    # Generate predictions with current and original models
    new_predict_dict = test(train_loader, model, device)
    original_predict_dict = test(train_loader, original_model, device)
    
    # Calculate MMD scores
    score_MMD = calculate_MMD(original_predict_dict, new_predict_dict)

    # Export anomaly scores for all videos (required for ablation study)
    export_anomaly_scores(score_MMD, original_predict_dict, new_predict_dict, 
                          video_labels, args, dataset_name)

    # Create updated labels based on MMD results
    new_video_labels = copy.deepcopy(video_labels)
    new_frame_labels = copy.deepcopy(frame_labels)
    updated_count = 0
    
    for video_name, p_value in score_MMD.items():
        # Only update abnormal videos meeting significance threshold
        if video_labels[video_name] == [1.0] and p_value < args.p_value:
            print(f"Video {video_name} meets threshold (p={p_value:.4f}) - updating label")
            new_video_labels[video_name] = [0.0]
            new_frame_labels[video_name] = np.zeros(len(frame_labels[video_name]))
            updated_count += 1
            
    print(f"Updated labels for {updated_count} videos based on MMD analysis")

    """Evaluate new labels against ground truth"""
    forget_class = '_'.join(sorted(args.forget_class))
    selected_videos_file = os.path.join(args.cur_path, f'{dataset_type}_dataset', dataset_name, f'{dataset_name}_train.txt')
    
    # Load true labels for comparison
    true_video_labels = load_pickle(os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', f'updated_{forget_class}_video_label.pickle'))
    true_frame_labels = load_pickle(os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', f'updated_{forget_class}_frame_label.pickle'))
    
    # Calculate metrics
    selected_videos = des_videos(selected_videos_file)
    print(f"Videos selected for evaluation: {len(selected_videos)}")
    
    # Calculate F1 score
    F1 = calculate_f1(new_video_labels, true_video_labels, selected_videos)
    print(f"Video-level F1 Score: {F1:.4f}")
    
    # Calculate frame-level TS_align
    video_ts_align = {}
    missing_count = 0
    
    for vid in selected_videos:
        if vid in new_frame_labels and vid in true_frame_labels:
            predict_labels = new_frame_labels[vid]
            true_labels = true_frame_labels[vid]
            
            ts_value = calculate_ts_align(predict_labels, true_labels)
            if ts_value is not None:
                video_ts_align[vid] = ts_value
            else:
                print(f"Video {vid}: TS_align calculation failed (insufficient frames)")
        else:
            missing_count += 1
            print(f"Warning: Video {vid} missing from predicted or true labels")
    
    # Only calculate average if we have valid results
    if video_ts_align:
        avg_ts_align = np.mean(list(video_ts_align.values()))
        print(f"Average TS_align: {avg_ts_align:.4f}")
    else:
        print("No valid TS_align values computed")
    
    print(f"Missing label data for {missing_count} videos")
    
    # Save updated labels
    with open(new_video_filename, 'wb') as f:
        pickle.dump(new_video_labels, f)

    with open(new_frame_filename, 'wb') as f:
        pickle.dump(new_frame_labels, f)

    print(f"Threshold filtering complete. Generated files:\n{new_video_filename}\n{new_frame_filename}")
    
    return new_video_filename, new_frame_filename


def export_anomaly_scores(score_MMD, original_predict_dict, new_predict_dict, 
                          video_labels, args, dataset_name):
    """
    Export anomaly scores for all videos to CSV file
    
    Args:
        score_MMD: Dictionary of MMD p-values for each video
        original_predict_dict: Predictions from original model
        new_predict_dict: Predictions from current model
        video_labels: Original video labels
        args: Command line arguments
        dataset_name: Name of the dataset
    """
    # Normalize dataset_type to remove trailing underscore if present
    dataset_type = args.dataset_type.rstrip('_') if hasattr(args, 'dataset_type') else 'shanghaitech'
    output_dir = os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', 'new_label')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for export
    data_list = []
    for video_name in score_MMD.keys():
        p_value = score_MMD[video_name]
        original_score_mean = np.mean(original_predict_dict[video_name])
        original_score_max = np.max(original_predict_dict[video_name])
        new_score_mean = np.mean(new_predict_dict[video_name])
        new_score_max = np.max(new_predict_dict[video_name])
        original_label = video_labels.get(video_name, [None])[0] if video_name in video_labels else None
        
        data_list.append({
            'video_id': video_name,
            'mmd_pvalue': p_value,
            'original_score_mean': original_score_mean,
            'original_score_max': original_score_max,
            'new_score_mean': new_score_mean,
            'new_score_max': new_score_max,
            'score_decrease_ratio_mean': new_score_mean / original_score_mean if original_score_mean > 0 else 0,
            'score_decrease_ratio_max': new_score_max / original_score_max if original_score_max > 0 else 0,
            'original_label': original_label,
            'is_abnormal': 1 if original_label == 1.0 else 0
        })
    
    # Save to CSV
    df = pd.DataFrame(data_list)
    csv_path = os.path.join(output_dir, f'anomaly_scores_{dataset_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Anomaly scores exported to: {csv_path}")
    
    # Also save to pickle for easy loading
    pickle_path = os.path.join(output_dir, f'anomaly_scores_{dataset_name}.pickle')
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_list, f)
    print(f"Anomaly scores also saved to pickle: {pickle_path}")


def run_ablation_study(args, dataset_name, pvalue_thresholds=None, use_score_decrease=True):
    """
    消融实验：测试不同pvalue阈值下的标签修正性能
    
    Args:
        args: Command line arguments
        dataset_name: Name of the dataset
        pvalue_thresholds: List of pvalue thresholds to test. If None, uses default range.
        use_score_decrease: Whether to use score decrease condition. If False, only uses pvalue threshold.
                           This allows testing the effect of pvalue threshold independently.
        
    Returns:
        Dictionary containing results for each threshold
    """
    print("\n" + "="*60)
    print("开始消融实验：测试不同pvalue阈值下的标签修正性能")
    print(f"使用分数下降条件: {use_score_decrease}")
    if use_score_decrease and hasattr(args, 'score_decrease_threshold'):
        print(f"分数下降阈值: {args.score_decrease_threshold}")
    print("="*60)
    
    # Normalize dataset_type to remove trailing underscore if present
    dataset_type = args.dataset_type.rstrip('_') if hasattr(args, 'dataset_type') and args.dataset_type else 'shanghaitech'
    
    # Debug: Print dataset type being used
    print(f"消融实验使用的数据集类型: {dataset_type}")
    print(f"数据集名称: {dataset_name}")
    
    # Load anomaly scores
    output_dir = os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', 'new_label')
    pickle_path = os.path.join(output_dir, f'anomaly_scores_{dataset_name}.pickle')
    print(f"查找异常分数文件的路径: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        print(f"错误：异常分数文件不存在: {pickle_path}")
        print("请先运行calculate_MMD_and_update_labels函数生成异常分数文件")
        return None
    
    with open(pickle_path, 'rb') as f:
        anomaly_scores = pickle.load(f)
    print(f"已加载 {len(anomaly_scores)} 个视频的异常分数")
    
    # Load original labels
    video_labels_path = os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', args.video_label_dict)
    frame_labels_path = os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', args.frame_label_dict)
    video_labels = load_pickle(video_labels_path)
    frame_labels = load_pickle(frame_labels_path)
    
    # Load true labels for evaluation
    forget_class = '_'.join(sorted(args.forget_class))
    true_video_labels = load_pickle(os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', f'updated_{forget_class}_video_label.pickle'))
    true_frame_labels = load_pickle(os.path.join(args.cur_path, f'{dataset_type}_dataset', 'GT', f'updated_{forget_class}_frame_label.pickle'))
    
    selected_videos_file = os.path.join(args.cur_path, f'{dataset_type}_dataset', dataset_name, f'{dataset_name}_train.txt')
    selected_videos = des_videos(selected_videos_file)
    
    # Default pvalue thresholds if not provided
    if pvalue_thresholds is None:
        pvalue_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    results = []
    
    for pvalue_thresh in pvalue_thresholds:
        print(f"\n测试 pvalue 阈值: {pvalue_thresh:.3f}")
        
        # Create updated labels based on current threshold
        new_video_labels = copy.deepcopy(video_labels)
        new_frame_labels = copy.deepcopy(frame_labels)
        updated_count = 0
        
        for item in anomaly_scores:
            video_name = item['video_id']
            p_value = item['mmd_pvalue']
            original_score_mean = item['original_score_mean']
            new_score_mean = item['new_score_mean']
            original_score_max = item['original_score_max']
            new_score_max = item['new_score_max']
            
            # Apply label correction if conditions are met
            pvalue_condition = p_value < pvalue_thresh
            score_condition = True
            if use_score_decrease and hasattr(args, 'score_decrease_threshold'):
                score_condition = (new_score_mean < original_score_mean * args.score_decrease_threshold and
                                 new_score_max < original_score_max * args.score_decrease_threshold)
            
            if (video_labels.get(video_name) == [1.0] and pvalue_condition and score_condition):
                new_video_labels[video_name] = [0.0]
                new_frame_labels[video_name] = np.zeros(len(frame_labels[video_name]))
                updated_count += 1
        
        print(f"  修正了 {updated_count} 个视频的标签")
        
        # Calculate F1 score
        F1 = calculate_f1(new_video_labels, true_video_labels, selected_videos)
        
        # Calculate frame-level TS_align
        video_ts_align = {}
        for vid in selected_videos:
            if vid in new_frame_labels and vid in true_frame_labels:
                predict_labels = new_frame_labels[vid]
                true_labels = true_frame_labels[vid]
                ts_value = calculate_ts_align(predict_labels, true_labels)
                if ts_value is not None:
                    video_ts_align[vid] = ts_value
        
        avg_ts_align = np.mean(list(video_ts_align.values())) if video_ts_align else None
        
        result = {
            'pvalue_threshold': pvalue_thresh,
            'updated_videos_count': updated_count,
            'f1_score': F1,
            'avg_ts_align': avg_ts_align,
            'valid_ts_align_count': len(video_ts_align)
        }
        results.append(result)
        
        print(f"  F1 Score: {F1:.4f}")
        ts_align_str = f"{avg_ts_align:.4f}" if avg_ts_align is not None else "N/A"
        print(f"  平均 TS_align: {ts_align_str}")
    
    # Save results to file
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, f'ablation_study_results_{dataset_name}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n消融实验结果已保存到: {results_path}")
    
    # Also save detailed JSON
    results_json_path = os.path.join(output_dir, f'ablation_study_results_{dataset_name}.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"详细结果（JSON格式）已保存到: {results_json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("消融实验总结")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    return results