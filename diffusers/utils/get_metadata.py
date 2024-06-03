import sys
sys.path.append('..')
sys.path.append('.')
import torch
import numpy as np
import random
import argparse
import json
import pickle

def main(args):
    # step 1: get the training feature
    with open(f'experiments/{args.model_type}/{args.method}/{args.method}_{args.model_type}_member_scores_all_steps.pth', 'rb') as f:
        member_scores = torch.load(f)
    with open(f'experiments/{args.model_type}/{args.method}/{args.method}_{args.model_type}_member_scores_all_steps.pth', 'rb') as f:
        nonmember_scores = torch.load(f)
    with open(f'experiments/{args.model_type}/{args.method}/{args.method}_{args.model_type}_sum_score_result.json', 'rb') as f:
        result = json.load(f)

    threshold = result['best_threshold_at_1_FPR']
    member_scores = member_scores.numpy()
    nonmember_scores = nonmember_scores.numpy()
    metadata = dict(member_scores=member_scores, nonmember_scores=nonmember_scores, threshold=threshold)
    print(type(metadata), type(member_scores),type(nonmember_scores), type(threshold) )
    with open(args.output + f'metadata_{args.method}_{args.model_type}.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    


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
    parser.add_argument('--output', type=str, default='../custom_nodes/assets/')
    parser.add_argument('--method', type=str, default='secmi', choices=['secmi', 'pia', 'pfami'])
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm', 'kohaku'], default='sd')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)