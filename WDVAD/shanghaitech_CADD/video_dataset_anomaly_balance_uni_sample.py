import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch


class dataset(Dataset):
    def __init__(self, args,new_video_filename=None,dataset_name="default_dataset", train=True, trainlist=None,updated_list=None, testlist=None):
        """
        :param args:
        self.dataset_path: path to dir contains anomaly datasets
        self.dataset_name: name of dataset which use now
        self.feature_modal: features from different input, contain rgb, flow or combine of above type
        self.feature_pretrain_model: the model name of feature extraction
        self.feature_path: the dir contain all features, use for training and testing
        self.videoname: videonames of dataset
        self.trainlist: videonames of dataset for training
        self.testlist: videonames of dataset for testing
        self.train: boolen type, if it is True, the dataset class return training data
        self.t_max: the max of sampling in training
        """
        self.args = args
        self.cur_path = args.cur_path
        self.dataset_type = args.dataset_type
        self.dataset_name = dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        self.new_video_filename = new_video_filename
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.cur_path, f'{self.dataset_type}_dataset' ,'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        elif self.feature_pretrain_model == 'i3d':
            self.feature_path = os.path.join(self.cur_path,f'{self.dataset_type}_dataset' , f'features_video', self.feature_pretrain_model, self.feature_modal)

        self.videoname = os.listdir(self.feature_path)
        self.data_dict = self.data_dict_creater()
        if updated_list:
            self.updated_list = self.txt2list(txtpath=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , self.dataset_name,updated_list))

        if trainlist:
            self.trainlist = self.txt2list(txtpath=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , self.dataset_name,trainlist))
        else:
            self.trainlist = self.txt2list(txtpath=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , self.dataset_name, f'{dataset_name}_train.txt'))
            
        if testlist:
            self.testlist = self.txt2list(txtpath=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , self.dataset_name,testlist))
        else:
            self.testlist = self.txt2list(txtpath=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , self.dataset_name, f'{dataset_name}_test.txt'))

        if new_video_filename is not None:
            self.video_label_dict = self.pickle_reader(
                file=new_video_filename)
        else:
            self.video_label_dict = self.pickle_reader(
                file=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' ,  'GT', 'video_label.pickle'))
        if updated_list:
            self.updated_list = self.txt2list(txtpath=os.path.join(self.cur_path, f'{self.dataset_type}_dataset' , self.dataset_name, 'updated_samples.txt'))
            self.normal_video_train = self.p_n_split_dataset_2(self.updated_list)
            _, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict, self.trainlist)
        else:
            self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict, self.trainlist)

        self.train = train
        self.t_max = args.max_seqlen



    def data_dict_creater(self):
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''): 
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict


    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train
    def p_n_split_dataset_2(self, updated_list):
        normal_video_train = []
        for t in updated_list:
            normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train


    def __getitem__(self, index):

        if self.args.larger_mem: #0
            if self.train:
                train_video_name = []
                start_index = []
                anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size) #shanghai:sample_size=30,avenue:10
                normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
                anomaly_features = torch.zeros(0)
                normaly_features = torch.zeros(0)
                for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                    anomaly_data_video_name = a_i.replace('\n', '').replace('Ped', 'ped')
                    normaly_data_video_name = n_i.replace('\n', '').replace('Ped', 'ped')
                    train_video_name += anomaly_data_video_name
                    train_video_name += normaly_data_video_name
                    anomaly_feature = self.data_dict[anomaly_data_video_name]
                    anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)
                    start_index += r
                    anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                    # shape = (1, seq_len, feature_dim )
                    normaly_feature = self.data_dict[normaly_data_video_name]
                    normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                    start_index += r
                    normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                    anomaly_features = torch.cat((anomaly_features, anomaly_feature),
                                                 dim=0)  # combine anomaly_feature of different a_i
                    normaly_features = torch.cat((normaly_features, normaly_feature),
                                                 dim=0)  # combine normaly_feature of different n_i
                # normaly_label = torch.zeros((self.args.sample_size, 1))
                # anomaly_label = torch.ones((self.args.sample_size, 1))
                if self.args.label_type == 'binary':
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)
                else:
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.zeros((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)

                return [anomaly_features, normaly_features], [anomaly_label, normaly_label], [train_video_name,start_index]
            else:
                data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
                self.feature = self.data_dict[data_video_name]
                return self.feature, data_video_name

        else:
            if self.train:
                anomaly_train_video_name = []
                normaly_train_video_name = []
                anomaly_start_index = []
                anomaly_len_index = []
                normaly_start_index = []
                normaly_len_index = []
                anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
                normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)
                anomaly_features = torch.zeros(0)
                normaly_features = torch.zeros(0)
                for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                    anomaly_data_video_name = a_i.replace('\n', '').replace('Ped', 'ped')
                    normaly_data_video_name = n_i.replace('\n', '').replace('Ped', 'ped')
                    anomaly_train_video_name.append(anomaly_data_video_name)
                    normaly_train_video_name.append(normaly_data_video_name)
                    anomaly_feature = np.load(
                        file=os.path.join(self.feature_path, anomaly_data_video_name, 'feature.npy'))
                    anomaly_len_index.append(anomaly_feature.shape[0])
                    anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)
                    anomaly_start_index.append(r)
                    anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                    normaly_feature = np.load(
                        file=os.path.join(self.feature_path, normaly_data_video_name, 'feature.npy'))
                    normaly_len_index.append(normaly_feature.shape[0])
                    normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                    normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)#【1,300,2048】
                    normaly_start_index.append(r)
                    anomaly_features = torch.cat((anomaly_features, anomaly_feature),
                                                 dim=0)  # combine anomaly_feature of different a_i
                    normaly_features = torch.cat((normaly_features, normaly_feature),
                                                 dim=0)  # combine normaly_feature of different n_i
                if self.args.label_type == 'binary':
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)
                elif self.args.label_type == 'unary':
                    normaly_label = torch.zeros((self.args.sample_size, 1))
                    anomaly_label = torch.ones((self.args.sample_size, 1))
                else:
                    normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                    anomaly_label = torch.cat((torch.zeros((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)

                train_video_name = anomaly_train_video_name + normaly_train_video_name
                start_index = anomaly_start_index + normaly_start_index
                len_index = anomaly_len_index + normaly_len_index

                return [anomaly_features, normaly_features], [anomaly_label, normaly_label], [train_video_name, start_index, len_index]
            else:
                data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
                self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
                return self.feature, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)

        else:
            return len(self.testlist)



            
