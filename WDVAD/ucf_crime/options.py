import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='AR_Net')

here = Path(__file__).resolve().parent

parser.add_argument('--cur_path', type=str, default=str(here), help='path to dir contains anomaly datasets')
parser.add_argument('--cid', type=int, required=True,
                    help='Client ID, should be an integer between 0 and NUM_CLIENTS')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--loss_type', default='DMIL_C', type=str,  help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')
parser.add_argument('--batch_size',  type=int, default=1, help='number of samples in one itration')
parser.add_argument('--sample_step', type=int, default=1, help='')
parser.add_argument('--feature_modal', type=str, default='combine', help='features from different input, options contain rgb, flow , combine')
parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750)')
parser.add_argument('--Lambda', type=str, default='1_20')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')
parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7')
parser.add_argument('--k', type=int, default=4, help='value of k')
parser.add_argument('--larger_mem', type=int, default=0, help='')
parser.add_argument('--label_type', type=str, default='unary')
parser.add_argument('--sample_size',  type=int, default=8, help='Normal or abnormal number of samples in one itration')
# parser.add_argument('--pretrained_ckpt_scene_1', default='cid_1_round_5.pkl', 
#                     help='Pretrained model checkpoint for client/scene 1')
# parser.add_argument('--pretrained_ckpt_scene_2', default='cid_2_round_5.pkl', 
#                     help='Pretrained model checkpoint for client/scene 2')
# parser.add_argument('--pretrained_ckpt_scene_3', default='CADD_3_iter_1000.pkl', 
#                     help='Pretrained model checkpoint for client/scene 3')
# parser.add_argument('--pretrained_ckpt_scene_4', default='CADD_4_iter_1000.pkl', 
#                     help='Pretrained model checkpoint for client/scene 4')
parser.add_argument('--server_address', type=str, default='localhost:8080',
                    help='gRPC server address ')
parser.add_argument('--sample_fraction', type=float, default=1.0,
                    help="Fraction of available clients used for fit/evaluate (default: 1.0)")
parser.add_argument('--min_num_clients', type=int, default=4,
                    help="Minimum number of available clients required for sampling (default: 2)")
parser.add_argument('--s', type=int, default=1, help='layer_idx')
parser.add_argument('--p', type=int, default=100, help="rand_percent")
parser.add_argument('--e', type=float, default=0.2, help="eta")
parser.add_argument('--b', type=float, default=0.1, help="beta")
parser.add_argument('--dataset_path', type=str, default='ucf_dataset', help="path to dataset")
parser.add_argument('--dataset_type', type=str, default='ucf_', help="ucf or others_")
parser.add_argument('--t', type=str, default=0.1, help="Train the weight until the standard deviation of the recorded "
                                            "losses is less than a given threshold. Default: 0.1")

parser.add_argument('--frame_label_dict', type=str, default="frame_label.pickle", 
                   help='Frame-level label file (pre-update, baseline annotations)') #换数据集时需要更改
parser.add_argument('--video_label_dict', type=str, default="video_label.pickle", 
                   help='Video-level label file (pre-update, baseline annotations)') #换数据集时需要更改

parser.add_argument('--t_id', type=list, default=[1],
                    help='Teacher client IDs excluded from training (e.g., [1] for client 1)')

parser.add_argument('--p_value', type=float, default=0.05, help='Significance threshold for statistical test (default: 0.05)')
parser.add_argument('--score_decrease_threshold', type=float, default=0.5, help='Threshold for score decrease in LSC (e.g., 0.5 means new score must be < 50% of old score)')

parser.add_argument('--pretrained_ckpt_scene_1', default='cid_1_round_1.pkl', 
                    help='Pretrained model checkpoint for client/scene 1') 
parser.add_argument('--pretrained_ckpt_scene_2', default='cid_2_round_5.pkl', 
                    help='Pretrained model checkpoint for client/scene 2') 

# 更新数据集时需要修改 forget_class
# 1
parser.add_argument('--forget_class', type=list, default=['Abuse','Arrest'],
                    help='Anomaly classes to be updated (single or multiple)')
# 2    
# parser.add_argument('--forget_class', type=list, default=['Burglary','Stealing'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 3            
# parser.add_argument('--forget_class', type=list, default=['Explosion','Fighting', 'Vandalism'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 4
# parser.add_argument('--forget_class', type=list, default=['RoadAccidents'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 5
# parser.add_argument('--forget_class', type=list, default=['Robbery'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 6            
# parser.add_argument('--forget_class', type=list, default=['Arson','Shoplifting'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 7            
# parser.add_argument('--forget_class', type=list, default=['Assault','Shooting'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 8
# parser.add_argument('--forget_class', type=list, default=['Abuse','Explosion', 'Shooting', 'Shoplifting'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 9
# parser.add_argument('--forget_class', type=list, default=['Arrest', 'Assault', 'Fighting', 'Vandalism'],
#                     help='Anomaly classes to be updated (single or multiple)')
# 10
# parser.add_argument('--forget_class', type=list, default=['Abuse', 'Arrest', 'Assault', 'Fighting', 'Vandalism'],
#                     help='Anomaly classes to be updated (single or multiple)')



