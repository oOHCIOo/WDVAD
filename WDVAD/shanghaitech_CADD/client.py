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
        pretrained_ckpt = os.path.join(self.cur_path,f'{self.dataset_type}_ckpt',f'ckpt_{self.forget_class}',ckpt_path)

        # Set label paths
        video_labels_path = os.path.join(self.cur_path,f'{self.dataset_type}_dataset' , 'GT',self.video_label_dict)
        frame_labels_path =  os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , 'GT',self.frame_label_dict)
        
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
        return self.get_parameters({}), result,{}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        self.rounds = config.get("server_round", 0)
        # Update student clients with global parameters
        if self.cid not in self.t_id:
            self.set_parameters(parameters)
            save_dir = os.path.join(self.cur_path,f'result_{self.dataset_name}','ckpt')
            os.makedirs(save_dir,exist_ok= True)
            save_file = os.path.join(save_dir,f'cid_{self.cid}_aggregate_{self.rounds}.pkl')
            torch.save(self.model.state_dict(),save_file)
        else:
            # Teacher clients always use their local models
            self.model = self.local_model
            
        # Perform adaptive aggregation for student clients after round 2
        if self.rounds >= 2 and self.cid not in self.t_id:
            print(f"Round {self.rounds}: Client {self.cid} performing adaptive aggregation")
            new_video_filename = os.path.join(self.args.cur_path,f'{self.dataset_type}_dataset','GT','new_label',f'new_video_label.pickle')
            train_data = dataset(args=self.args, new_video_filename=new_video_filename, dataset_name=self.dataset_name, train=True)
            self.global_model = copy.deepcopy(self.model)
            self.local_initialization(train_data)
            self.model = copy.deepcopy(self.local_model)
            save_file_2 = os.path.join(save_dir,f'cid_{self.cid}_local_model_round_{self.rounds}.pkl')
            torch.save(self.model.state_dict(),save_file_2)
        else:
            print(f"Round {self.rounds}: Client {self.cid} skipping adaptive aggregation")
            
        # Perform inference with current model
        label_dict_path = os.path.join(self.cur_path,f'{self.dataset_type}_dataset', 'GT')
        predict_dict = test(test_loader=self.val_loader, model=self.model, device=self.device)

        if not predict_dict:
            raise ValueError("predict_dict does not exist")

        # Load appropriate label version based on client type and round
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
        
        # Process model outputs for evaluation metrics
        all_predict_np = np.zeros(0)
        all_label_np = np.zeros(0)
        normal_predict_np = np.zeros(0)
        normal_label_np = np.zeros(0)
        abnormal_predict_np = np.zeros(0)
        abnormal_label_np = np.zeros(0)
        
        for k, v in predict_dict.items():
            # Process normal videos
            if video_label_dict[k] == [1.]:
                frame_labels = frame_label_dict[k]
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))])) 
                abnormal_predict_np = np.concatenate((abnormal_predict_np, v.repeat(16))) 
                abnormal_label_np = np.concatenate((abnormal_label_np, frame_labels[:len(v.repeat(16))]))  
            # Process abnormal videos
            elif video_label_dict[k] == [0.]:
                frame_labels = frame_label_dict[k]
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
                normal_predict_np = np.concatenate((normal_predict_np, v.repeat(16)))  
                normal_label_np = np.concatenate((normal_label_np, frame_labels[:len(v.repeat(16))]))  

        # Calculate evaluation metrics
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
        binary_all_predict_np = scorebinary(all_predict_np, threshold=0.5)
        tn, fp, fn, tp = confusion_matrix(y_true=all_label_np, y_pred=binary_all_predict_np).ravel()
        all_ano_false_alarm = fp / (fp + tn)
        
        binary_normal_predict_np = scorebinary(normal_predict_np, threshold=0.5)
        fp_n = binary_normal_predict_np.sum()
        normal_count = normal_label_np.shape[0]
        normal_ano_false_alarm = fp_n / normal_count

        abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
        binary_abnormal_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
        tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abnormal_predict_np).ravel()
        abnormal_ano_false_alarm = fp / (fp + tn)
        
        # Visualize anomaly maps
        save_path = os.path.join(self.cur_path, f'result_{self.dataset_name}', 'score_test')
        anomap(predict_dict, label_dict=frame_label_dict, save_path=save_path, round=self.rounds)

        print(f"AUC: {all_auc_score:.4f}, FAR: {all_ano_false_alarm:.4f}")
        
        # Save results to file
        file_name = f'result_cid_{self.cid}.txt'
        file_path = os.path.join(self.cur_path,f'result_{self.dataset_name}', file_name)
        save_dir = os.path.dirname(file_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        try:
            with open(file_path, 'a+') as f:
                f.write(f"\nround_num:{self.rounds}\n")
                f.write(f'round_{self.rounds}_AUC_Score_all_video: {all_auc_score}\n')
                f.write(f'round_{self.rounds}_AUC_Score_abnormal_video: {abnormal_auc_score}\n')
                f.write(f'round_{self.rounds}_ano_false_alarm_all_video: {all_ano_false_alarm}\n')
                f.write(f'round_{self.rounds}_ano_false_alarm_normal_video: {normal_ano_false_alarm}\n')
                f.write(f'round_{self.rounds}_ano_false_alarm_abnormal_video: {abnormal_ano_false_alarm}\n')
            print(f"Results saved to: {file_path}")
        except Exception as e:
            print(f"Failed to write file! Path: {file_path}, Error: {str(e)}")

        return float(0), len(self.val_loader.dataset), {"AUC": float(all_auc_score)}

def main():
    args = options.parser.parse_args()
    # print(args)
    if args.dataset_type == 'shanghaitech':
        NUM_CLIENTS = 2
    elif args.dataset_type == 'CADD' and args.min_num_clients == 2:
        NUM_CLIENTS = 2
    elif args.dataset_type == 'CADD' and args.min_num_clients == 4:
        NUM_CLIENTS = 4
    else:
        print("Please define the number of clients for this dataset_type in client.py")
        return
    assert args.cid <= NUM_CLIENTS
    dataset_name = f"{args.dataset_type}_{args.cid}"
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