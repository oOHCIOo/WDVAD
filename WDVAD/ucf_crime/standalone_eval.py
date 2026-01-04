import torch
import numpy as np
import os
import pickle
import re
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
from torch.utils.data import DataLoader

# Import from existing project files
import options
from model import Model_single
from video_dataset_anomaly_balance_uni_sample import dataset
from test import test
from utils import scorebinary


# ====================== 新增：TS align 计算函数 ======================
def compute_ts_align(frame_scores: np.ndarray, frame_labels: np.ndarray) -> float:
    """
    TS align = 1 - mean(|e_i - e_{i-1}|), 其中 e_i = |l_i - t_i|
    l_i 使用帧级预测分数 (0~1)，t_i 为帧级真值标签 (0/1)。
    做法：对每个视频单独计算，再对所有视频取均值。
    """
    scores = np.asarray(frame_scores, dtype=float).flatten()
    labels = np.asarray(frame_labels, dtype=float).flatten()
    n = min(len(scores), len(labels))
    if n <= 1:
        return np.nan
    e = np.abs(scores[:n] - labels[:n])
    return 1.0 - np.mean(np.abs(e[1:] - e[:-1]))
# ==================================================================


def evaluate_model(args):
    """Loads a model and evaluates it on the test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = Model_single(n_feature=2048).to(device)
    try:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint)
        print(f"Model weights loaded successfully from {args.ckpt_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    model.eval()

    # 2. Load Data
    try:
        val_dataset = dataset(args=args, dataset_name=args.dataset_name, train=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)
        print(f"Validation loader created with {len(val_loader.dataset)} samples.")
    except Exception as e:
        print(f"Error creating dataset/loader: {e}")
        return

    # 3. Get Predictions
    print("Running model on the test set...")
    predict_dict, length_dict = test(test_loader=val_loader, model=model, device=device)
    if not predict_dict:
        print("Error: The 'test' function returned an empty prediction dictionary.")
        return

    # 4. Load Ground Truth Labels
    try:
        frame_label_dict = pickle.load(open(args.frame_label_path, 'rb'))
        video_label_dict = pickle.load(open(args.video_label_path, 'rb'))
        print(f"Labels loaded from {args.frame_label_path} and {args.video_label_path}")
    except Exception as e:
        print(f"Error loading label files: {e}")
        return

    # 5. Calculate Metrics (Copied and adapted from client.py)
    print("Calculating metrics...")
    all_predict_list, all_label_list = [], []
    normal_predict_list = []
    abnormal_predict_list, abnormal_label_list = [], []
    per_class_data = {}

    for k, v in predict_dict.items():
        valid_length = length_dict.get(k)
        if valid_length is None:
            print(f"Warning: Video '{k}' not found in length_dict, skipped.")
            continue

        # v is already a 1D numpy array (num_clips,) from test.py
        # Repeat each clip score 16 times to get frame-level scores
        frame_scores = np.repeat(v, 16)
        # Slice to the actual number of frames for the video
        predictions = frame_scores[:valid_length]

        # Re-insert the definition of labels
        frame_labels = frame_label_dict.get(k)
        labels = (np.zeros(valid_length) if frame_labels is None else frame_labels[:valid_length]).flatten()

        all_predict_list.append(predictions)
        all_label_list.append(labels)

        video_label = video_label_dict.get(k, [0.0])
        if video_label == [1.0]:
            abnormal_predict_list.append(predictions)
            abnormal_label_list.append(labels)

            match = re.match(r'([A-Za-z]+)', k)
            if match:
                class_name = match.group(1)
                if class_name not in per_class_data:
                    per_class_data[class_name] = {'preds': [], 'labels': []}
                per_class_data[class_name]['preds'].append(predictions)
                per_class_data[class_name]['labels'].append(labels)
        else:
            normal_predict_list.append(predictions)

    if not all_predict_list:
        print("Error: No data to evaluate, all test videos were skipped.")
        return

    all_predict_np = np.concatenate(all_predict_list)
    all_label_np = np.concatenate(all_label_list)

    # --- Overall Metrics ---
    all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np) if len(np.unique(all_label_np)) > 1 else 0.0
    abnormal_auc_score = 0.0
    if abnormal_predict_list:
        abnormal_predict_np = np.concatenate(abnormal_predict_list)
        abnormal_label_np = np.concatenate(abnormal_label_list)
        abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np) if len(np.unique(abnormal_label_np)) > 1 else 0.0

    cm = confusion_matrix(y_true=all_label_np, y_pred=scorebinary(all_predict_np, threshold=0.5))
    all_ano_false_alarm = (cm[0, 1] / (cm[0, 0] + cm[0, 1])) if cm.size == 4 and (cm[0, 0] + cm[0, 1]) > 0 else 0.0

    normal_ano_false_alarm = 0.0
    if normal_predict_list:
        normal_predict_np = np.concatenate(normal_predict_list)
        normal_ano_false_alarm = np.sum(scorebinary(normal_predict_np, threshold=0.5)) / len(normal_predict_np)

    abnormal_ano_false_alarm = 0.0
    if abnormal_predict_list and np.concatenate(abnormal_label_list).size > 0:
        cm_a = confusion_matrix(y_true=np.concatenate(abnormal_label_list), y_pred=scorebinary(np.concatenate(abnormal_predict_list), threshold=0.5))
        abnormal_ano_false_alarm = (cm_a[0, 1] / (cm_a[0, 0] + cm_a[0, 1])) if cm_a.size == 4 and (cm_a[0, 0] + cm_a[0, 1]) > 0 else 0.0

    # ===== 新增：整体 F1 分数（Threshold=0.5） =====
    overall_f1 = f1_score(all_label_np, scorebinary(all_predict_np, threshold=0.5), zero_division=0)

    # --- Per-Class Metrics ---
    per_class_auc_scores = {}
    for class_name, data in per_class_data.items():
        if not data['preds']:
            continue
        class_preds_np = np.concatenate(data['preds'])
        class_labels_np = np.concatenate(data['labels'])
        if len(np.unique(class_labels_np)) > 1:
            class_auc = roc_auc_score(y_true=class_labels_np, y_score=class_preds_np)
            per_class_auc_scores[class_name] = class_auc
        else:
            per_class_auc_scores[class_name] = 0.0

    # ================================== 新增代码开始 ==================================
    # --- Per-Class Average Frame-Level Metrics (based on threshold) ---
    per_class_intermediate_metrics = {}
    threshold = 0.5  # 设定分类阈值

    # 额外：为“整体 TS”准备容器（按视频计算再求平均）
    ts_all_video_scores = []

    # 遍历所有视频，计算单个视频的指标并按类别存储；同时计算该视频的 TS
    for video_name, clip_scores in predict_dict.items():
        match = re.match(r'([A-Za-z]+)', video_name)
        if not match:
            continue
        class_name = match.group(1)

        valid_length = length_dict.get(video_name)
        frame_labels_raw = frame_label_dict.get(video_name)

        if valid_length is None or frame_labels_raw is None:
            continue

        labels = frame_labels_raw[:valid_length]

        if not isinstance(clip_scores, np.ndarray):
            clip_scores = clip_scores.cpu().numpy()
        frame_predictions_scores = np.repeat(clip_scores, 16)[:valid_length]

        # ---- 新增：计算该视频 TS 并收集 ----
        ts_val = compute_ts_align(frame_predictions_scores, labels)
        if not np.isnan(ts_val):
            ts_all_video_scores.append(ts_val)

        frame_predictions_binary = (frame_predictions_scores >= threshold).astype(int)

        accuracy = accuracy_score(labels, frame_predictions_binary)
        precision = precision_score(labels, frame_predictions_binary, zero_division=0)
        recall = recall_score(labels, frame_predictions_binary, zero_division=0)
        f1 = f1_score(labels, frame_predictions_binary, zero_division=0)

        if class_name not in per_class_intermediate_metrics:
            per_class_intermediate_metrics[class_name] = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []
            }

        per_class_intermediate_metrics[class_name]['accuracy'].append(accuracy)
        per_class_intermediate_metrics[class_name]['precision'].append(precision)
        per_class_intermediate_metrics[class_name]['recall'].append(recall)
        per_class_intermediate_metrics[class_name]['f1_score'].append(f1)

    # 计算每个类别的平均分数
    final_per_class_avg_metrics = {}
    for class_name, metrics_lists in per_class_intermediate_metrics.items():
        final_per_class_avg_metrics[class_name] = {
            'avg_accuracy': np.mean(metrics_lists['accuracy']),
            'avg_precision': np.mean(metrics_lists['precision']),
            'avg_recall': np.mean(metrics_lists['recall']),
            'avg_f1_score': np.mean(metrics_lists['f1_score']),
            'video_count': len(metrics_lists['accuracy'])
        }

    # ===== 新增：整体 TS（按视频均值） =====
    ts_align_all = np.nanmean(ts_all_video_scores) if ts_all_video_scores else 0.0
    # ================================== 新增代码结束 ==================================


    # 6. Print Results
    print("\n--- Evaluation Results ---")
    print(f"AUC Score (All Videos): {all_auc_score:.6f}")
    print(f"AUC Score (Abnormal Videos Only): {abnormal_auc_score:.6f}")
    print("___")
    print(f"False Alarm Rate (All Videos): {all_ano_false_alarm:.6f}")
    print(f"False Alarm Rate (Normal Videos Only): {normal_ano_false_alarm:.6f}")
    print(f"False Alarm Rate (Abnormal portions of abnormal videos): {abnormal_ano_false_alarm:.6f}")

    # ===== 新增：整体 F1 与整体 TS 输出 =====
    print("\n--- Overall Metrics (Threshold=0.5) ---")
    print(f"Overall F1 Score: {overall_f1:.6f}")
    print(f"TS Align (All Videos): {ts_align_all:.6f}")

    print("\n--- Per-Class Frame-Level AUC ---")
    if per_class_auc_scores:
        for class_name, auc in sorted(per_class_auc_scores.items()):
            print(f"  {class_name}: {auc:.6f}")
    else:
        print("  No abnormal classes found in the provided data.")

    # 打印每个类别的平均帧级指标
    print("\n--- Per-Class Average Frame-Level Metrics (Threshold=0.5) ---")
    if final_per_class_avg_metrics:
        for class_name, avg_metrics in sorted(final_per_class_avg_metrics.items()):
            print(f"Class: {class_name} (based on {avg_metrics['video_count']} videos)")
            print(f"  Avg Accuracy:  {avg_metrics['avg_accuracy']:.4f}")
            print(f"  Avg Precision: {avg_metrics['avg_precision']:.4f}")
            print(f"  Avg Recall:    {avg_metrics['avg_recall']:.4f}")
            print(f"  Avg F1-Score:  {avg_metrics['avg_f1_score']:.4f}")
            print("_" * 30)
    else:
        print("  No data to calculate per-class average metrics.")

if __name__ == "__main__":
    # Add new arguments for the standalone script
    parser = options.parser
   
    # 1
    parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-1/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Abuse_Arrest_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Abuse_Arrest_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
   
    # 2
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-2/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Burglary_Stealing_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Burglary_Stealing_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
   
    # 3
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-3/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Explosion_Fighting_Vandalism_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Explosion_Fighting_Vandalism_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
   
    # 4
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-4/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_RoadAccidents_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_RoadAccidents_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
    
    # 5
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-5/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Robbery_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Robbery_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
   
    # 6
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-6/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Arson_Shoplifting_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Arson_Shoplifting_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
  
    # 7
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-7/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Assault_Shooting_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Assault_Shooting_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
  
    # 8
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-8/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Abuse_Explosion_Shooting_Shoplifting_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Abuse_Explosion_Shooting_Shoplifting_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
  
    # 9
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-9/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Arrest_Assault_Fighting_Vandalism_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Arrest_Assault_Fighting_Vandalism_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
  
    # 10
    # parser.add_argument('--ckpt_path', type=str, default='result_ucf_2-10/ckpt/cid_2_local_model_round_5.pkl', help='Path to the model checkpoint (.pkl) file.')
    # parser.add_argument('--frame_label_path', type=str, default='ucf_dataset/GT/updated_Abuse_Arrest_Assault_Fighting_Vandalism_frame_label.pickle', help='Path to the frame-level ground truth (.pickle) file.')
    # parser.add_argument('--video_label_path', type=str, default='ucf_dataset/GT/updated_Abuse_Arrest_Assault_Fighting_Vandalism_video_label.pickle', help='Path to the video-level ground truth (.pickle) file.')
    # parser.add_argument('--dataset_name', type=str, default='ucf_2', help='Name of the dataset partition to test on, e.g., ucf_2')
   
   
    args = parser.parse_args()
   
    # ============ 新增部分：保存输出到 txt 文件 ============
    first_part = re.split(r'[\\/]', args.ckpt_path, maxsplit=1)[0]
    log_path = f"{first_part}.txt"
    sys.stdout = open(log_path, "w", encoding="utf-8")
    sys.stderr = sys.stdout  # 同时捕获错误输出
    # ====================================================

    evaluate_model(args)

    # ============ 恢复输出 ============
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"所有输出已保存到 {log_path}")
    # =================================

