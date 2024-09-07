import sys
sys.path.append('..')
sys.path.append('.')
import tqdm
import torch
import numpy as np
import random
import os
import argparse
import json,time
from accelerate import Accelerator
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from copymark import load_dataset, benchmark, test

def compute_50th_score(member_scores, nonmember_scores):
    '''
        member_scores: [N, S]
        nonmember_scores: [N, S]
    '''
    return member_scores[:, 14], nonmember_scores[:, 14]

def compute_corr_score(member_scores, nonmember_scores):
    '''
        member_scores: [N, S]
        nonmember_scores: [N, S]
    '''
    all_scores = torch.cat([member_scores, nonmember_scores], dim=0)    # [2N, S]
    timestep_mean = all_scores.mean(dim=0)  # [S]

    member_scores = (member_scores / timestep_mean).mean(dim=1) # [N]
    nonmember_scores = (nonmember_scores / timestep_mean).mean(dim=1)

    return member_scores, nonmember_scores

def compute_sum_score(member_scores, nonmember_scores):
    '''
        member_scores: [N, S]
        nonmember_scores: [N, S]
    '''
    return member_scores.sum(dim=1), nonmember_scores.sum(dim=1)

def main(args):
    # step 1: secmi and secmi++
    with open(f'experiments/{args.model_type}/secmi/secmi_{args.model_type}_member_scores_all_steps_test.pth', 'rb') as f:
        member_scores_all_steps = torch.load(f)
    with open(f'experiments/{args.model_type}/secmi/secmi_{args.model_type}_nonmember_scores_all_steps_test.pth', 'rb') as f:
        nonmember_scores_all_steps = torch.load(f)
    threshold_path = args.threshold_root + f'{args.model_type}/secmi/'

    member_scores_50th_step, nonmember_scores_50th_step = compute_50th_score(member_scores_all_steps, nonmember_scores_all_steps)
    member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
    test(member_scores_50th_step, nonmember_scores_50th_step, f'secmi_{args.model_type}_50th_score', args.output, threshold_path)
    test(member_corr_scores, nonmember_corr_scores, f'secmi_{args.model_type}_corr_score', args.output, threshold_path)

    # step 2: pia
    with open(f'experiments/{args.model_type}/pia/pia_{args.model_type}_member_scores_all_steps_test.pth', 'rb') as f:
        member_scores_all_steps = torch.load(f)
    with open(f'experiments/{args.model_type}/pia/pia_{args.model_type}_nonmember_scores_all_steps_test.pth', 'rb') as f:
        nonmember_scores_all_steps = torch.load(f)
    member_sum_scores, nonmember_sum_scores = compute_sum_score(member_scores_all_steps, nonmember_scores_all_steps)
    member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
    threshold_path = args.threshold_root + f'{args.model_type}/pia/'
    test(member_sum_scores, nonmember_sum_scores, f'pia_{args.model_type}_sum_score', args.output, threshold_path)
    test(member_corr_scores, nonmember_corr_scores, f'pia_{args.model_type}_corr_score', args.output, threshold_path)

    # step 3: pfami
    with open(f'experiments/{args.model_type}/pfami/pfami_{args.model_type}_member_scores_all_steps_test.pth', 'rb') as f:
        member_scores_all_steps = torch.load(f)
    with open(f'experiments/{args.model_type}/pfami/pfami_{args.model_type}_nonmember_scores_all_steps_test.pth', 'rb') as f:
        nonmember_scores_all_steps = torch.load(f)
    member_sum_scores, nonmember_sum_scores = compute_sum_score(member_scores_all_steps, nonmember_scores_all_steps)
    member_corr_scores, nonmember_corr_scores = compute_corr_score(member_scores_all_steps, nonmember_scores_all_steps)
    threshold_path = args.threshold_root + f'{args.model_type}/pfami/'
    test(member_sum_scores, nonmember_sum_scores, f'pfami_{args.model_type}_sum_score', args.output, threshold_path)
    test(member_corr_scores, nonmember_corr_scores, f'pfami_{args.model_type}_corr_score', args.output, threshold_path)
    
    
    


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='datasets/', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='outputs/')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--use-ddp', type=bool, default=False)
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--threshold-root', type=str, default='experiments/')
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm', 'kohaku', 'laion_mi'], default='laion_mi')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)