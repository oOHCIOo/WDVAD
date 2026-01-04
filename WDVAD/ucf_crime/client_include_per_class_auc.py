import warnings
from collections import OrderedDict
import argparse
import flwr as fl
import torch
from video_dataset_anomaly_balance_uni_sample import dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from train import train
import os
import pickle
import options
from test import test
import copy
from utils import scorebinary
from model import Model_single
from utils import anomap
from SALA import SALA

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Federated Learning Client')

warnings.filterwarnings("ignore", category=UserWarning)

class FlowerClient(fl.client.NumPyClient):
    """A FlowerClient that trains a model for anomaly detection in federated setting."""

    def __init__(self, dataset_name,val_loader, args) -> None:
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.val_loader = val_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize model based on client type (teacher/student)
        if args.cid in args.t_id:
            self.model = Model_single(n_feature=2048).to(self.device)
            self.local_model = copy.deepcopy(self.model)
        else:
            self.model = Model_single(n_feature=2048).to(self.device)
            self.local_model = copy.deepcopy(self.model)
            self.global_model = Model_single(n_feature=2048).to(self.device)
        self.cur_path = args.cur_path
        self.save_path = os.path.join(args.cur_path,f'result_{self.dataset_name}')
        self.forget_class='_'.join(sorted(args.forget_class))
        self.cid = args.cid
        self.t_id=args.t_id
        self.video_label_dict = args.video_label_dict
        self.frame_label_dict = args.frame_label_dict
        self.pretrained_ckpt_scene_1 = args.pretrained_ckpt_scene_1
        self.pretrained_ckpt_scene_2 = args.pretrained_ckpt_scene_2
        self.dataset_type = args.dataset_type
        self.dataset_path = args.dataset_path
        # self.pretrained_ckpt_scene_3 = args.pretrained_ckpt_scene_3
        # self.pretrained_ckpt_scene_4 = args.pretrained_ckpt_scene_4
        self.batch_size = args.batch_size
        # Initialize adaptive learning module
        self.SALA = SALA(cid=self.args.cid, batch_size=1, rand_percent=self.args.p,
                         layer_idx=self.args.s, eta=self.args.e, beta=self.args.b, args=self.args,
                         threshold=self.args.t)

    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def local_initialization(self,train_data):
        """Perform adaptive local aggregation"""
        self.SALA.adaptive_local_aggregation(train_data,self.global_model, self.local_model)

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        # Get global round number from server
        self.rounds = config.get("server_round", 0)
        # Update student clients with global parameters
        if self.cid not in self.t_id:
            self.set_parameters(parameters)
        # Read hyperparameters from server configuration
        epochs =  config["epochs"]
        print(f"Client {self.cid} received server round: {self.rounds}")
        
        # Configure pretrained model path based on scene
        scene_id = self.cid 
        if self.cid==1:
            ckpt_path=self.pretrained_ckpt_scene_1
        elif self.cid==2:
            ckpt_path=self.pretrained_ckpt_scene_2
        # elif self.cid==3:
        #     ckpt_path=self.pretrained_ckpt_scene_3
        # elif self.cid==4:
        #     ckpt_path=self.pretrained_ckpt_scene_4
        print(f"Pretrained model path for scene {scene_id}: {ckpt_path}")
        pretrained_ckpt = os.path.join(self.cur_path,f'{self.dataset_type}ckpt',f'ckpt_{self.forget_class}',ckpt_path)

        # Set label paths
        video_labels_path = os.path.join(self.cur_path,f'{self.dataset_type}dataset' , 'GT',self.video_label_dict)
        frame_labels_path =  os.path.join(self.cur_path, f'{self.dataset_type}dataset' , 'GT',self.frame_label_dict)
        
        # Use updated local model after round 2
        if self.rounds > 2:
            self.model = copy.deepcopy(self.local_model)
            
        # Perform local training
        result=train(round_num=self.rounds,save_path=self.save_path,model=self.model,cid=self.cid, 
                     args=self.args, dataset_name=self.dataset_name,epochs=epochs, device=self.device,
                     pretrained_ckpt=pretrained_ckpt,video_labels_path=video_labels_path, 
                     frame_labels_path=frame_labels_path)
        self.local_model = copy.deepcopy(self.model)
        
        # Return updated parameters and statistics
        return self.get_parameters({}), result, {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.rounds = config.get("server_round", 0)
        
        # Preserve federated learning logic for model loading and saving
        if self.cid not in self.t_id:
            self.set_parameters(parameters)
            save_dir = os.path.join(self.cur_path,f'result_{self.dataset_name}','ckpt')
            os.makedirs(save_dir,exist_ok= True)
            save_file = os.path.join(save_dir,f'cid_{self.cid}_aggregate_{self.rounds}.pkl')
            torch.save(self.model.state_dict(),save_file)
        else:
            # Teacher clients always use their local models
            self.model = self.local_model
            
        # Preserve adaptive aggregation logic
        if self.rounds >= 2 and self.cid not in self.t_id:
            print(f"Round {self.rounds}: Client {self.cid} performing adaptive aggregation")
            new_video_filename = os.path.join(self.args.cur_path,f'{self.dataset_type}dataset','GT','new_label',f'new_video_label.pickle')
            train_data = dataset(args=self.args, new_video_filename=new_video_filename, dataset_name=self.dataset_name, train=True)
            self.global_model = copy.deepcopy(self.model)
            self.local_initialization(train_data)
            self.model = copy.deepcopy(self.local_model)
            save_file_2 = os.path.join(save_dir,f'cid_{self.cid}_local_model_round_{self.rounds}.pkl')
            torch.save(self.model.state_dict(),save_file_2)
        else:
            print(f"Round {self.rounds}: Client {self.cid} skipping adaptive aggregation")
            
        # --- Start of Robust Evaluation Logic ---
        label_dict_path = os.path.join(self.cur_path,f'{self.dataset_type}dataset', 'GT')
        predict_dict, length_dict = test(test_loader=self.val_loader, model=self.model, device=self.device)

        if not predict_dict:
            raise ValueError("predict_dict does not exist")

        # Preserve label loading logic based on round and client type
        if self.cid in self.t_id:
            frame_label_dict = pickle.load(open(os.path.join(label_dict_path, f'updated_{self.forget_class}_frame_label.pickle'), 'rb'))
            video_label_dict = pickle.load(open(os.path.join(label_dict_path, f'updated_{self.forget_class}_video_label.pickle'), 'rb'))
            print(f"Test: Client_{self.cid} using updated labels at round {self.rounds}")
        else:
            if self.rounds == 1:
                frame_label_dict = pickle.load(open(os.path.join(label_dict_path, f'frame_label.pickle'), 'rb'))
                video_label_dict = pickle.load(open(os.path.join(label_dict_path, f'video_label.pickle'), 'rb'))
                print(f"Test: Client_{self.cid} using original labels at round {self.rounds}") 
            else:
                frame_label_dict = pickle.load(open(os.path.join(label_dict_path, f'updated_{self.forget_class}_frame_label.pickle'), 'rb'))
                video_label_dict = pickle.load(open(os.path.join(label_dict_path, f'updated_{self.forget_class}_video_label.pickle'), 'rb'))
                print(f"Test: Client_{self.cid} using updated labels at round {self.rounds}")
        
        if not frame_label_dict:
            raise ValueError("frame_label_dict does not exist")
        
        # Robust data processing and metric calculation
        all_predict_list, all_label_list = [], []
        normal_predict_list = []
        abnormal_predict_list, abnormal_label_list = [], []
        per_class_data = {} # For per-class AUC

        for k, v in predict_dict.items():
            valid_length = length_dict.get(k)
            if valid_length is None:
                print(f"Warning: Video '{k}' not found in length_dict, skipped.")
                continue

            predictions = v.repeat(16)[:valid_length]
            
            frame_labels = frame_label_dict.get(k)
            labels = np.zeros(valid_length) if frame_labels is None else frame_labels[:valid_length]
            
            all_predict_list.append(predictions)
            all_label_list.append(labels)

            video_label = video_label_dict.get(k, [0.0])
            if video_label == [1.0]:
                abnormal_predict_list.append(predictions)
                abnormal_label_list.append(labels)
                
                # Group predictions and labels by class by parsing the video name
                import re
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
            return float(0), len(self.val_loader.dataset), {"AUC": 0.0}

        all_predict_np = np.concatenate(all_predict_list)
        all_label_np = np.concatenate(all_label_list)

        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np) if len(np.unique(all_label_np)) > 1 else 0.0
        
        if abnormal_predict_list:
            abnormal_predict_np = np.concatenate(abnormal_predict_list)
            abnormal_label_np = np.concatenate(abnormal_label_list)
            abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np) if len(np.unique(abnormal_label_np)) > 1 else 0.0
        else:
            abnormal_auc_score = 0.0
            abnormal_predict_np = np.array([])

        # --- Start of Per-Class AUC Calculation ---
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
        # --- End of Per-Class AUC Calculation ---

        cm = confusion_matrix(y_true=all_label_np, y_pred=scorebinary(all_predict_np, threshold=0.5))
        all_ano_false_alarm = (cm[0, 1] / (cm[0, 0] + cm[0, 1])) if cm.size == 4 and (cm[0, 0] + cm[0, 1]) > 0 else 0.0

        if normal_predict_list:
            normal_predict_np = np.concatenate(normal_predict_list)
            normal_ano_false_alarm = np.sum(scorebinary(normal_predict_np, threshold=0.5)) / len(normal_predict_np)
        else:
            normal_ano_false_alarm = 0.0
        
        abnormal_ano_false_alarm = 0.0
        if abnormal_predict_np.size > 0:
            cm_a = confusion_matrix(y_true=np.concatenate(abnormal_label_list), y_pred=scorebinary(abnormal_predict_np, threshold=0.5))
            abnormal_ano_false_alarm = (cm_a[0, 1] / (cm_a[0, 0] + cm_a[0, 1])) if cm_a.size == 4 and (cm_a[0, 0] + cm_a[0, 1]) > 0 else 0.0

        save_path = os.path.join(self.cur_path, f'result_{self.dataset_name}', 'score_test')
        anomap(predict_dict, label_dict=frame_label_dict, length_dict=length_dict, save_path=save_path, round=self.rounds)

        print(f"AUC: {all_auc_score:.4f}, FAR: {all_ano_false_alarm:.4f}")
        
        file_name = f'result_cid_{self.cid}.txt'
        file_path = os.path.join(self.cur_path,f'result_{self.dataset_name}', file_name)
        try:
            with open(file_path, 'a+') as f:
                f.write(f"\nround_num:{self.rounds}\n")
                f.write(f'round_{self.rounds}_AUC_Score_all_video: {all_auc_score}\n')
                f.write(f'round_{self.rounds}_AUC_Score_abnormal_video: {abnormal_auc_score}\n')
                f.write(f'round_{self.rounds}_ano_false_alarm_all_video: {all_ano_false_alarm}\n')
                f.write(f'round_{self.rounds}_ano_false_alarm_normal_video: {normal_ano_false_alarm}\n')
                f.write(f'round_{self.rounds}_ano_false_alarm_abnormal_video: {abnormal_ano_false_alarm}\n')
                
                # Add per-class AUC scores to the file
                f.write(f'\n--- Per-Class Frame-Level AUC ---\n')
                if per_class_auc_scores:
                    for class_name, auc in sorted(per_class_auc_scores.items()):
                        f.write(f'round_{self.rounds}_AUC_Score_{class_name}: {auc}\n')
                else:
                    f.write('No per-class scores calculated for this round.\n')

            print(f"Results saved to: {file_path}")
        except Exception as e:
            print(f"Failed to write file! Path: {file_path}, Error: {str(e)}")

        return float(0), len(self.val_loader.dataset), {"AUC": float(all_auc_score)}

def main():
    args = options.parser.parse_args()
    # print(args)
    if args.dataset_type == 'ucf_':
        NUM_CLIENTS = 2
    else:
        print("Please define the number of clients for this dataset_type in client.py")
        return
    assert args.cid <= NUM_CLIENTS
    dataset_name = f"{args.dataset_type}{args.cid}"
    vallist = f'{dataset_name}_test.txt'
    
    # Create validation dataset
    valdataset = dataset(args=args, dataset_name=dataset_name, train=False, testlist=vallist)
    val_loader = DataLoader(
                dataset=valdataset, batch_size=1, pin_memory=True,
                num_workers=0, shuffle=False
            )
    print(f"val_loader contains {len(val_loader.dataset)} samples")

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient( dataset_name=dataset_name, val_loader=val_loader, args=args).to_client()
    )

if __name__ == "__main__":
    main()