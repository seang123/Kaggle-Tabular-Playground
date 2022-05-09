
import numpy as np
import pandas as pd


pred = np.load("mlp_predictions_test.npy")
try:
    pred = np.squeeze(pred, axis=-1)
except ValueError:
    print("no dim of size 1")
print(pred.shape)


sub = pd.read_csv('Data/sample_submission.csv')
sub['target'] = pred
sub.to_csv("mlp_submission_pseudo_labels.csv", index=False)


