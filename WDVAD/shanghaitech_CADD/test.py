import torch
import pickle
import options
import os
# import torch.nn.functional as F
# import utils
import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

def test(test_loader, model, device): 
    result = {}
    for i, data in enumerate(test_loader):
        feature, data_video_name = data
        feature = feature.to(device)
        with torch.no_grad():
                element_logits = model(feature, is_training=False)
        element_logits = element_logits.cpu().data.numpy().reshape(-1)
        result[data_video_name[0]] = element_logits
    return result

def scorebinary(scores=None, threshold=0.5):
    scores_threshold = scores.copy()
    scores_threshold[scores_threshold < threshold] = 0
    scores_threshold[scores_threshold >= threshold] = 1
    return scores_threshold





