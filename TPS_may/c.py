import numpy as np
import pandas as pd


df = pd.read_csv("mlp_submission.csv")
print(df.shape)

t = df['target'].values

t0 = np.where(t < 0.000001)[0]
t1 = np.where(t > 0.999999)[0]

print(len(t0))
print(len(t1))


