import argparse
import itertools
import torch
import math
import os
import re
import sys
import getpass
import random
import pickle
import subprocess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from optim import get_optimizer, AdamInverseSqrtWithWarmup
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import Sampler
from logger import create_logger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' % getpass.getuser()

def initialize_exp(params):
    '''
    Initialize the experience
    - dump parameters
    - create a logger
    '''
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    #get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    #check experiment name
    assert len(params.exp_name.strip()) > 0

    #create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    params.save_path = os.path.join(params.dump_path, 'checkpoint.pth')
    params.save_best = os.path.join(params.dump_path, 'Best_model.pth')
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger

def get_dump_path(params):
    '''
    Create a directory to store the experiment.
    '''
    dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    #create the sweep path
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()
    
    #create a job ID
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()

def bool_flag(s):
    '''
    Parse boolean arguments from the command line.
    '''
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

def combination(iterable, r):
    '''
    combine the negative sample with binary tuple
    '''
    pool = list(iterable)
    n = len(pool)
    for indices in itertools.permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield list(pool[i] for i in indices)

def label_equal_mask(all_labels, label):
    '''
    mask all the equal label
    '''
    mask = [0] * len(all_labels)
    for index, i in enumerate(all_labels):
        if label.mul(i).sum(0) != 0:
            mask[index] = 1        
    
    return mask

def get_triplets(labels):
    '''
    get triplet for training
    '''
    triplets = []
    for label in labels:
        label_mask = np.matmul(labels, np.transpose(label)) > 0
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            continue
        negative_indices = np.where(np.logical_not(label_mask))[0]
        anchor_positives = list(combination(label_indices, 2))  # All anchor-positive pairs

        # Add all negatives for all positive pairs
        temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                         for neg_ind in negative_indices]
        triplets += temp_triplets

    return torch.LongTensor(np.array(triplets))

def triplet_hashing_loss(image_embedding, text_embedding, labels, margin=1):
    
    triplets = get_triplets(labels)

    ap_distances_i_i = (image_embedding[triplets[:, 0]] - image_embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances_i_i = (image_embedding[triplets[:, 0]] - image_embedding[triplets[:, 2]]).pow(2).sum(1)
    ap_distances_t_t = (text_embedding[triplets[:, 0]] - text_embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances_t_t = (text_embedding[triplets[:, 0]] - text_embedding[triplets[:, 2]]).pow(2).sum(1)
    ap_distances_i_t = (image_embedding[triplets[:, 0]] - text_embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances_i_t = (image_embedding[triplets[:, 0]] - text_embedding[triplets[:, 2]]).pow(2).sum(1)
    ap_distances_t_i = (text_embedding[triplets[:, 0]] - image_embedding[triplets[:, 1]]).pow(2).sum(1)
    an_distances_t_i = (text_embedding[triplets[:, 0]] - image_embedding[triplets[:, 2]]).pow(2).sum(1)

    losses_i_i = F.relu(ap_distances_i_i - an_distances_i_i + margin)
    losses_t_t = F.relu(ap_distances_t_t - an_distances_t_t + margin)
    losses_i_t = F.relu(ap_distances_i_t - an_distances_i_t + margin)
    losses_t_i = F.relu(ap_distances_t_i - an_distances_t_i + margin)

    return losses_i_i.mean(), losses_t_t.mean(), losses_i_t.mean(), losses_t_i.mean()

def cal_result(data_loarder, Image_encoder, Text_encoder, params):
    binary_code_image = []
    binary_code_text = []
    total_labels = []
    with torch.no_grad():
        for images, tokens, segment, mask, labels in data_loarder:
            total_labels.append(labels)
            output_image = Image_encoder(images)
            output_text = Text_encoder(tokens, segment, mask)
            binary_code_image.append(output_image.data.cpu())
            binary_code_text.append(output_text.data.cpu())

    return torch.sign(torch.cat(binary_code_image)), torch.sign(torch.cat(binary_code_text)), torch.cat(total_labels)

def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    Ns = Ns.type(torch.FloatTensor)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = ((trn_label[query_result]*query_label).sum(1) > 0).float()
        P = torch.cumsum(correct.type(torch.FloatTensor), dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP

def save_model(Image_encoder, Text_encoder, save_path):
    model_weights = {'image_encoder': Image_encoder.state_dict(),
                    'text_encoder': Text_encoder.state_dict()}
    torch.save(model_weights, save_path)

def load_model(Image_encoder, Text_encoder, save_path):
    model_weights = torch.load(save_path)
    Image_encoder.load_state_dict(model_weights['image_encoder'])
    Text_encoder.load_state_dict(model_weights['text_encoder'])

def optimize(optimizer, parameters, params, loss):
    optimizer.zero_grad()
    loss.backward()
    if params.clip_grad_norm > 0:
        clip_grad_norm_(parameters, params.clip_grad_norm)
    optimizer.step()

def eval_all(database, querys, Image_encoder, Text_encoder, params, logger):
    #caculate map
    Image_encoder.eval()
    Text_encoder.eval()

    train_binary_image, train_binary_text, train_labels = cal_result(database, Image_encoder, Text_encoder, params)
    test_binary_image, test_binary_text, test_labels = cal_result(querys, Image_encoder, Text_encoder, params)

    mAP_i_t = float(compute_mAP(train_binary_text, test_binary_image, train_labels, test_labels))
    logger.info('mAP_image_text: %f' % mAP_i_t)

    mAP_t_i = float(compute_mAP(train_binary_image, test_binary_text, train_labels, test_labels))
    logger.info('mAP_text_image: %f' % mAP_t_i)

    return mAP_i_t, mAP_t_i
