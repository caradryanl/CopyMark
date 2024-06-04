import sys
sys.path.append('..')
sys.path.append('.')
import torch
import numpy as np
import random
import argparse
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def remove_outliers(data, threshold=3):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

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

def main(args):

    # Initialize a list to store the data for all methods
    all_data = []

    for method in ['secmi', 'secmi++', 'pia', 'pfami', 'gsa_1', 'gsa_2']:
        # Step 1: get the training feature
        if method != 'secmi++':
            with open(f'experiments/{args.model_type}/{method}/{method}_{args.model_type}_member_scores_all_steps.pth', 'rb') as f:
                member_scores = torch.load(f)
            with open(f'experiments/{args.model_type}/{method}/{method}_{args.model_type}_nonmember_scores_all_steps.pth', 'rb') as f:
                nonmember_scores = torch.load(f)
            with open(f'experiments/{args.model_type}/{method}/{method}_{args.model_type}_sum_score_result.json', 'rb') as f:
                result = json.load(f)
        else:
            with open(f'experiments/{args.model_type}/secmi/secmi_{args.model_type}_member_scores_all_steps.pth', 'rb') as f:
                member_scores = torch.load(f)
            with open(f'experiments/{args.model_type}/secmi/secmi_{args.model_type}_nonmember_scores_all_steps.pth', 'rb') as f:
                nonmember_scores = torch.load(f)
            with open(f'experiments/{args.model_type}/secmi/secmi_{args.model_type}_corr_score_result.json', 'rb') as f:
                result = json.load(f)
            member_scores, nonmember_scores = compute_corr_score(member_scores, nonmember_scores)

        threshold = result['best_threshold_at_1_FPR']
        member_scores = member_scores.numpy()
        nonmember_scores = nonmember_scores.numpy()

        if method == 'secmi':
            member_scores = member_scores[:, 14]
            nonmember_scores = nonmember_scores[:, 14]
        elif method == 'pfami' or method == 'pia':
            member_scores = member_scores.sum(axis=-1)
            nonmember_scores = nonmember_scores.sum(axis=-1)

        # Remove outliers
        member_scores = remove_outliers(member_scores)
        nonmember_scores = remove_outliers(nonmember_scores)

        # Ensure both arrays have the same length after outlier removal
        min_length = min(len(member_scores), len(nonmember_scores))
        member_scores = member_scores[:min_length]
        nonmember_scores = nonmember_scores[:min_length]

        if method == 'pia':
            member_scores, nonmember_scores = member_scores/100000, nonmember_scores/100000

        data = pd.DataFrame({
            'member': member_scores,
            'non-member': nonmember_scores,
        })

        data_melted = data.melt(var_name='Category', value_name='Scores')
        data_melted['Method'] = method
        data_melted['Threshold'] = threshold/100000 if method == 'pia' else threshold
        all_data.append(data_melted)

    # Set the style and context for larger font sizes and grid lengths
    # sns.set(style="whitegrid")
    # sns.set_context("talk", font_scale=1.4)

    # Plot the data
    fig = plt.figure(figsize=(108, 12))  # Set the overall figure siz
    mapping = {
        'pia': 'PIA',
        'secmi++': 'SecMI++',
        'secmi': 'SecMI',
        'pfami': 'PFAMI',
        'gsa_1': 'GSA1',
        'gsa_2': 'GSA2',
    }
    colors = ['palegreen', 'gold']

    for i, method_data in enumerate(all_data):
        method = method_data['Method'].iloc[0]
        threshold = method_data['Threshold'].iloc[0]

        _, ax = plt.subplots(figsize=(8, 8))  # Create a new figure for each method

        categories = ['member', 'non-member']
        data_to_plot = [method_data[method_data['Category'] == cat]['Scores'].values for cat in categories]
        boxprops = dict(linewidth=3)
        medianprops = dict(linewidth=3, color='lightgray')
        whiskerprops = dict(linewidth=3)
        capprops = dict(linewidth=3)
        flierprops = dict(marker='o', color='green', markersize=12, alpha=1, markeredgewidth=2)

        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=categories, widths=0.6,
                        boxprops=boxprops, medianprops=medianprops,
                        whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
        

        # Change the color of each box
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        ax.set_title(f'{mapping[method]}', fontsize=43)
        ax.set_ylabel('Scores', fontsize=32)
        ax.tick_params(axis='y', labelsize=32)
        ax.tick_params(axis='x', labelsize=35)

        ax.yaxis.grid(True, linestyle='--', linewidth=1)
        ax.legend(fontsize=30, loc='upper left')

        plt.subplots_adjust(left=0.23, right=0.99, top=0.88, bottom=0.12, wspace=0.4)  # Adjust subplot parameters
        plt.show()
        



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