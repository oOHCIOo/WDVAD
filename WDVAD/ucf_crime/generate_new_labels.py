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
    output_dir = os.path.join(args.dataset_path, 'GT', 'new_label')
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
    
    # Export detailed prediction scores for comparison
    export_detailed_predictions(original_predict_dict, new_predict_dict, 
                                  video_labels, args, dataset_name)


def export_detailed_predictions(original_predict_dict, new_predict_dict, 
                                  video_labels, args, dataset_name):
    """
    导出每个视频在两个模型下的详细预测分数，用于对比分析
    
    Args:
        original_predict_dict: 原始模型的预测结果 {video_id: scores_array}
        new_predict_dict: 新模型的预测结果 {video_id: scores_array}
        video_labels: 原始视频标签
        args: 命令行参数
        dataset_name: 数据集名称
    """
    output_dir = os.path.join(args.dataset_path, 'GT', 'new_label')
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_data = []
    
    # 获取所有视频ID
    all_video_ids = set(original_predict_dict.keys()) & set(new_predict_dict.keys())
    
    for video_id in sorted(all_video_ids):
        original_scores = original_predict_dict[video_id]
        new_scores = new_predict_dict[video_id]
        
        # 转换为numpy数组以便处理
        if not isinstance(original_scores, np.ndarray):
            original_scores = np.array(original_scores)
        if not isinstance(new_scores, np.ndarray):
            new_scores = np.array(new_scores)
        
        original_label = video_labels.get(video_id, [None])[0] if video_id in video_labels else None
        
        # 计算各种统计量
        original_mean = np.mean(original_scores)
        original_std = np.std(original_scores)
        original_min = np.min(original_scores)
        original_max = np.max(original_scores)
        original_median = np.median(original_scores)
        
        new_mean = np.mean(new_scores)
        new_std = np.std(new_scores)
        new_min = np.min(new_scores)
        new_max = np.max(new_scores)
        new_median = np.median(new_scores)
        
        # 计算差异
        mean_diff = new_mean - original_mean
        max_diff = new_max - original_max
        relative_change_mean = (mean_diff / original_mean * 100) if original_mean != 0 else 0
        relative_change_max = (max_diff / original_max * 100) if original_max != 0 else 0
        
        # 检查分数是否完全相同
        scores_identical = np.array_equal(original_scores, new_scores) if len(original_scores) == len(new_scores) else False
        
        detailed_data.append({
            'video_id': video_id,
            'original_label': original_label,
            'is_abnormal': 1 if original_label == 1.0 else 0,
            'num_clips': len(original_scores),
            # 原始模型统计
            'original_mean': original_mean,
            'original_std': original_std,
            'original_min': original_min,
            'original_max': original_max,
            'original_median': original_median,
            # 新模型统计
            'new_mean': new_mean,
            'new_std': new_std,
            'new_min': new_min,
            'new_max': new_max,
            'new_median': new_median,
            # 差异统计
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'relative_change_mean_pct': relative_change_mean,
            'relative_change_max_pct': relative_change_max,
            # 是否相同
            'scores_identical': scores_identical,
            # 保留原始分数数组（用于详细分析）
            'original_scores_array': original_scores.tolist(),
            'new_scores_array': new_scores.tolist()
        })
    
    # 保存详细数据到CSV（不包含数组，因为CSV不支持）
    df_detailed = pd.DataFrame([{k: v for k, v in item.items() 
                                 if k not in ['original_scores_array', 'new_scores_array']} 
                                for item in detailed_data])
    detailed_csv_path = os.path.join(output_dir, f'detailed_predictions_{dataset_name}.csv')
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"详细预测分数已导出到: {detailed_csv_path}")
    
    # 保存完整数据（包含数组）到pickle和JSON
    detailed_pickle_path = os.path.join(output_dir, f'detailed_predictions_{dataset_name}.pickle')
    with open(detailed_pickle_path, 'wb') as f:
        pickle.dump(detailed_data, f)
    print(f"完整详细数据（包含分数数组）已保存到: {detailed_pickle_path}")
    
    # 统计信息
    identical_count = sum(1 for item in detailed_data if item['scores_identical'])
    abnormal_count = sum(1 for item in detailed_data if item['is_abnormal'] == 1)
    abnormal_identical = sum(1 for item in detailed_data 
                            if item['is_abnormal'] == 1 and item['scores_identical'])
    
    print(f"\n预测分数对比统计:")
    print(f"  总视频数: {len(detailed_data)}")
    print(f"  分数完全相同的视频数: {identical_count} ({identical_count/len(detailed_data)*100:.1f}%)")
    print(f"  异常视频总数: {abnormal_count}")
    if abnormal_count > 0:
        print(f"  异常视频中分数完全相同的: {abnormal_identical} ({abnormal_identical/abnormal_count*100:.1f}%)")
    else:
        print(f"  异常视频中分数完全相同的: {abnormal_identical}")
    
    # 异常视频的分数变化统计
    abnormal_videos = [item for item in detailed_data if item['is_abnormal'] == 1]
    if abnormal_videos:
        mean_changes = [item['relative_change_mean_pct'] for item in abnormal_videos]
        print(f"\n异常视频的分数变化统计 (相对变化百分比):")
        print(f"  均值: {np.mean(mean_changes):.2f}%")
        print(f"  标准差: {np.std(mean_changes):.2f}%")
        print(f"  最小值: {np.min(mean_changes):.2f}%")
        print(f"  最大值: {np.max(mean_changes):.2f}%")
        print(f"  分数下降的视频数 (变化 < 0): {sum(1 for c in mean_changes if c < 0)}")
        print(f"  分数上升的视频数 (变化 > 0): {sum(1 for c in mean_changes if c > 0)}")


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
    # Verify models are distinct
    if model is original_model:
        print("Warning: Current model and original model are the same instance!")
    print(f"Model ID: {id(model)}, Original Model ID: {id(original_model)}")
    print(f"Weight difference: {torch.sum(model.fc.weight != original_model.fc.weight).item()}")
    
    # Load original labels
    video_labels = load_pickle(video_labels_path)
    frame_labels = load_pickle(frame_labels_path)

    def calculate_MMD(original_predict_dict, new_predict_dict, device):
        """
        Compute MMD (Maximum Mean Discrepancy) between original and new predictions
        
        Args:
            original_predict_dict: Predictions from original model
            new_predict_dict: Predictions from current model
            device: The device to perform computation on.
            
        Returns:
            Dictionary of MMD p-values for each video
        """
        score_MMD = {}
        common_video_ids = list(set(new_predict_dict.keys()) & set(original_predict_dict.keys()))
        total_videos = len(common_video_ids)
        print(f"Calculating MMD for {total_videos} common videos...")

        for i, video_id in enumerate(common_video_ids):
            # Prepare data for MMD calculation
            X = original_predict_dict[video_id].reshape(-1, 1)
            Y = new_predict_dict[video_id].reshape(-1, 1)
            
            # Compute MMD and p-value on the specified device
            _, p_value = permutation_test_mmd(X, Y, device, sigma=1.0, num_permutations=1000)
            score_MMD[video_id] = p_value

            # Add progress indicator for every video
            print(f"    ...processed video {i + 1} / {total_videos}: {video_id}")
        
        print(f"MMD calculation complete for all {total_videos} videos.")
        return score_MMD

    # Create output directory for new labels
    output_dir = os.path.join(args.dataset_path, 'GT', 'new_label')
    os.makedirs(output_dir, exist_ok=True)
    new_video_filename = os.path.join(output_dir, 'new_video_label.pickle')
    new_frame_filename = os.path.join(output_dir, 'new_frame_label.pickle')
    
    # Return existing files if already generated
    if os.path.exists(new_video_filename) and os.path.exists(new_frame_filename):
        print(f"Label files already exist, returning:\nVideo labels: {new_video_filename}\nFrame labels: {new_frame_filename}")
        return new_video_filename, new_frame_filename

    # Generate predictions with current and original models
    new_predict_dict, _ = test(train_loader, model, device)
    original_predict_dict, _ = test(train_loader, original_model, device)
    
    # Calculate MMD scores
    score_MMD = calculate_MMD(original_predict_dict, new_predict_dict, device)

    # Export anomaly scores for all videos
    export_anomaly_scores(score_MMD, original_predict_dict, new_predict_dict, 
                          video_labels, args, dataset_name)

    # Create updated labels based on MMD results
    new_video_labels = copy.deepcopy(video_labels)
    new_frame_labels = copy.deepcopy(frame_labels)
    updated_count = 0
    
    for video_name, p_value in score_MMD.items():
        # Only update abnormal videos meeting significance threshold
        # Only update abnormal videos if MMD is significant AND new model score is significantly lower
        if (video_labels.get(video_name) == [1.0] and p_value < args.p_value):
    #     if (video_labels.get(video_name) == [1.0] and 
    # p_value < args.p_value and 
    # np.mean(new_predict_dict[video_name]) < np.mean(original_predict_dict[video_name]) * args.score_decrease_threshold and
    # np.max(new_predict_dict[video_name]) < np.max(original_predict_dict[video_name]) * args.score_decrease_threshold):
            print(f"Video {video_name} meets threshold (p={p_value:.4f}) and score decreased (threshold={args.score_decrease_threshold}) - updating label")
            new_video_labels[video_name] = [0.0]
            new_frame_labels[video_name] = np.zeros(len(frame_labels[video_name]))
            updated_count += 1
    print(f"Updated labels for {updated_count} videos based on MMD analysis")

    """Evaluate new labels against ground truth"""
    forget_class = '_'.join(sorted(args.forget_class))
    selected_videos_file = os.path.join('ucf_dataset', dataset_name, f'{dataset_name}_train.txt')
    
    # Load true labels for comparison
    true_video_labels = load_pickle(os.path.join(args.dataset_path, 'GT', f'updated_{forget_class}_video_label.pickle'))
    true_frame_labels = load_pickle(os.path.join(args.dataset_path, 'GT', f'updated_{forget_class}_frame_label.pickle'))
    
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
    if use_score_decrease:
        print(f"分数下降阈值: {args.score_decrease_threshold}")
    print("="*60)
    
    # Load anomaly scores
    output_dir = os.path.join(args.dataset_path, 'GT', 'new_label')
    pickle_path = os.path.join(output_dir, f'anomaly_scores_{dataset_name}.pickle')
    
    if not os.path.exists(pickle_path):
        print(f"错误：异常分数文件不存在: {pickle_path}")
        print("请先运行calculate_MMD_and_update_labels函数生成异常分数文件")
        return None
    
    with open(pickle_path, 'rb') as f:
        anomaly_scores = pickle.load(f)
    print(f"已加载 {len(anomaly_scores)} 个视频的异常分数")
    
    # Load original labels
    video_labels_path = os.path.join(args.dataset_path, 'GT', args.video_label_dict)
    frame_labels_path = os.path.join(args.dataset_path, 'GT', args.frame_label_dict)
    video_labels = load_pickle(video_labels_path)
    frame_labels = load_pickle(frame_labels_path)
    
    # Load true labels for evaluation
    forget_class = '_'.join(sorted(args.forget_class))
    true_video_labels = load_pickle(os.path.join(args.dataset_path, 'GT', f'updated_{forget_class}_video_label.pickle'))
    true_frame_labels = load_pickle(os.path.join(args.dataset_path, 'GT', f'updated_{forget_class}_frame_label.pickle'))
    
    selected_videos_file = os.path.join('ucf_dataset', dataset_name, f'{dataset_name}_train.txt')
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
            if use_score_decrease:
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