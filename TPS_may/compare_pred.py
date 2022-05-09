import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


mlp_preds = pd.read_csv(f'mlp_submission.csv')['target'].values
#mlp_preds = np.around(mlp_preds, 3)

mlp_preds2 = pd.read_csv(f'mlp_submission2.csv')['target'].values
#mlp_preds2 = np.around(mlp_preds, 3)

xgb_preds = pd.read_csv('xgb_submission.csv')['target'].values
#xgb_preds = np.around(xgb_preds, 3)

tree_preds = pd.read_csv('dtree_submission.csv')['target'].values

rf_preds  = pd.read_csv('rand_forest_submission.csv')['target'].values

def compare():
    print(" MLP - MLP2 - XGB - dtree - rand for - avg ")
    for i in range(40):
        avg = np.mean(np.array([mlp_preds, mlp_preds2, xgb_preds, tree_preds, rf_preds]), axis=0)
        print(f"{mlp_preds[i]:.3f} - {mlp_preds2[i]:.3f}  - {xgb_preds[i]:.3f} - {tree_preds[i]:.3f} - {rf_preds[i]:.3f} - {avg[i]:.3f}")

def generate_submission(mlp_preds, xgb_preds):
    sub = []
    for i in range(mlp_preds.shape[0]):
        mp = abs(mlp_preds[i]-0.5)
        xp = abs(xgb_preds[i]-0.5)
        if mp > xp:
            sub.append(mlp_preds[i])
        elif mp < xp:
            sub.append(xgb_preds[i])
        else:
            sub.append(xgb_preds[i])
    return sub

def generate_submission_mean(preds: list):
    """ Take average prediction """
    print("Generating submission by average")
    temp = np.array(preds)
    sub = np.mean(temp, axis=0)
    return sub

def generate_submission_vote(preds: list):
    """ Simple majority wins vote """
    print("Generating submission by vote")
    preds = np.array(preds)
    n_voters = preds.shape[0]

    out = []
    for i in range(preds.shape[1]):
        v0 = 0
        v1 = 0
        for v in range(n_voters):
            if preds[v][i] <= 0.5:
                v0 += 1
            else:
                v1 += 1
        if v0 >= v1:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)


compare()

sub = generate_submission_mean([mlp_preds2, mlp_preds, xgb_preds, tree_preds, rf_preds])

sub2 = pd.read_csv("Data/sample_submission.csv")
sub2['target'] = sub
sub2.to_csv("ensemble_submission.csv", index=False)
