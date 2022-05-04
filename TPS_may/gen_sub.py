
import numpy as np
import pandas as pd


pred = np.squeeze(np.load("mlp_predictions.npy"), axis=-1)
print(pred.shape)


sub = pd.read_csv('Data/sample_submission.csv')
sub['target'] = pred
sub.to_csv("mlp_submission.csv", index=False)


