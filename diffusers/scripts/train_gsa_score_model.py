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

from stable_copyright import load_dataset, benchmark, test

def normalize(array, min=None, max=None):
    eps = 1e-5
    if min is not None and max is not None:
        min_val = min
        max_val = max
    else:
        min_val = np.min(array)
        max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val + eps)
    normalized_array = (normalized_array - 0.5) / 0.5
    return normalized_array, min_val, max_val

def preprocess(member, non_member, min=None, max=None):
    member = member[0:non_member.shape[0]]
    member_y_np = np.zeros(member.shape[0])
    nonmember_y_np = np.ones(non_member.shape[0])
    x = np.vstack((member, non_member))
    x_max, x_min, x_avg = x[np.isfinite(x)].max(), x[np.isfinite(x)].min(), x[np.isfinite(x)].mean()
    x = np.nan_to_num(x, nan=x_avg, posinf=x_max, neginf=x_min) # deal with exploding gradients

    x, min, max = normalize(x, min, max)
    y = np.concatenate((member_y_np, nonmember_y_np))

    def has_nan_inf(array):
        has_nan = np.isnan(array).any()
        has_inf = np.isinf(array).any()
        return has_nan or has_inf
    
    print(has_nan_inf(x))
    return x, y, min, max

def train_model(x, y):
    
    # xgb = SGDClassifier()
    # model = XGBClassifier(n_estimators=200)
    # model = RandomForestRegressor(n_estimators=200, random_state=42)
    # model = MLPClassifier(solver='lbfgs',
    #                 hidden_layer_sizes=(400, 200, 100, 50), random_state=42)
    model = XGBRegressor(n_estimators=50, max_depth=2)
    model.fit(x, y)

    def get_batches(X, y, batch_size):
        n_samples = X.shape[0]
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            yield X[start:end], y[start:end]
    
    # Training the model incrementally
    # for X_batch, y_batch in get_batches(x, y, batch_size=100):
    #     xgb.partial_fit(X_batch, y_batch, classes=np.unique(y))

    # y_pred = batch_predict(xgb, x, batch_size=100)
    y_pred = model.predict(x)

    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])
    # print(member_scores[0:10], nonmember_scores[0:10])
    return model, member_scores, nonmember_scores

def batch_predict(model, X, batch_size):
    n_samples = X.shape[0]
    predictions = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        X_batch = X[start:end]
        batch_predictions = model.predict(X_batch)
        predictions.extend(batch_predictions)
    return np.array(predictions)

def test_model(model, x, y):

    # y_pred = batch_predict(xgb, x, batch_size=100)
    y_pred = model.predict(x)
    member_scores = torch.tensor(y_pred[y <= 0.5])
    nonmember_scores = torch.tensor(y_pred[y > 0.5])

    # print(member_scores[0:10], nonmember_scores[0:10])

    return member_scores, nonmember_scores

def main(args):
    # step 1: get the training feature
    with open(f'experiments/{args.model_type}/gsa_{args.gsa_mode}/gsa_{args.gsa_mode}_{args.model_type}_features.npy', 'rb') as f:
        features = np.load(f)
    with open(f'experiments/{args.model_type}/gsa_{args.gsa_mode}/gsa_{args.gsa_mode}_{args.model_type}_features_test.npy', 'rb') as f:
        features_test = np.load(f)
    
    data_size = len(features)//2
    member_features, nonmember_features = features[0 :data_size], features[data_size:]
    member_features_test, nonmember_features_test = features_test[0 :data_size], features_test[data_size:]
    # print(f"member: {len(member_features)}, nonmember: {len(nonmember_features)}")       

    # step 2: train the model
    x, y, x_min, x_max = preprocess(member_features, nonmember_features)
    model, member_scores, nonmember_scores = train_model(x, y)

    metadata = dict(x=x, y=y, x_min=x_min, x_max=x_max, model=model)
    with open(args.output + f'metadata_gsa_{args.gsa_mode}_{args.model_type}.pkl', 'wb') as f:
        pickle.dump(metadata, f)


    TP = (member_scores <= 0.5).sum()
    TN = (nonmember_scores > 0.5).sum()
    FP = (nonmember_scores <= 0.5).sum()
    FN = (member_scores > 0.5).sum()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    print(f"gsa_{args.gsa_mode},{args.model_type}, training set tpr: {TPR}, fpr: {FPR}")

    # test the trained xgboost
    x_test, y_test, _, _ = preprocess(member_features_test, nonmember_features_test, x_min, x_max)
    member_scores, nonmember_scores = test_model(model, x_test, y_test)
    print(member_scores.max(), nonmember_scores.max())

    with open(f'experiments/{args.model_type}/gsa_{args.gsa_mode}/gsa_{args.gsa_mode}_{args.model_type}_member_scores_all_steps.pth', 'wb') as f:
        torch.save(member_scores, f)
    with open(f'experiments/{args.model_type}/gsa_{args.gsa_mode}/gsa_{args.gsa_mode}_{args.model_type}_nonmember_scores_all_steps.pth', 'wb') as f:
        torch.save(nonmember_scores, f)

    TP = (member_scores <= 0.5).sum()
    TN = (nonmember_scores > 0.5).sum()
    FP = (nonmember_scores <= 0.5).sum()
    FN = (member_scores > 0.5).sum()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    print(f"gsa_{args.gsa_mode},{args.model_type}, test set tpr: {TPR}, fpr: {FPR}")

    # tsne = TSNE(n_components=2, random_state=42)
    # # x_tsne, y_tsne = np.concatenate([x, x_test], axis=0), np.concatenate([y, y_test + 2], axis=0)
    # x_tsne, y_tsne = x_test, y_test
    # x_tsne = tsne.fit_transform(x_tsne)
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_tsne, cmap='viridis')
    # plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    # plt.title("t-SNE visualization of Synthetic Data")
    # plt.xlabel("t-SNE feature 1")
    # plt.ylabel("t-SNE feature 2")
    # plt.show()
    
    
    


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
    parser.add_argument('--gsa-mode', type=int, default=1, choices=[1, 2])
    parser.add_argument('--model-type', type=str, choices=['sd', 'sdxl', 'ldm', 'kohaku'], default='sd')
    args = parser.parse_args()

    fix_seed(args.seed)

    main(args)