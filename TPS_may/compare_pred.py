import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


mlp_preds = pd.read_csv(f'mlp_submission_poly.csv')['target'].values
mlp_preds = np.around(mlp_preds, 3)

xgb_preds = pd.read_csv('xgb_submission_poly.csv')['target'].values
xgb_preds = np.around(xgb_preds, 3)


def compare():
    for i in range(20):
        print(f"{mlp_preds[i]:.3f} - {xgb_preds[i]:.3f}")

    count = 0
    for i in range(mlp_preds.shape[0]):
        if mlp_preds[i] < 0.5 and xgb_preds[i] > 0.5:
            #print(f"{mlp_preds[i]:.3f} - {xgb_preds[i]:.3f} | {abs(mlp_preds[i] - 0.5):.3f} - {abs(xgb_preds[i] - 0.5):.3f}")
            count += 1
        elif mlp_preds[i] > 0.5 and xgb_preds[i] < 0.5:
            #print(f"{mlp_preds[i]:.3f} - {xgb_preds[i]:.3f} | {abs(mlp_preds[i] - 0.5):.3f} - {abs(xgb_preds[i] - 0.5):.3f}")
            count += 1
    print(count)

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

compare()

sub = generate_submission(mlp_preds, xgb_preds)

sub2 = pd.read_csv("Data/sample_submission.csv")
sub2['target'] = sub
sub2.to_csv("ensemble_submission.csv", index=False)
