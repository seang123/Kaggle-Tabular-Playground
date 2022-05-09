import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import time
from itertools import groupby
from operator import itemgetter

"""

Things we can compute:
    how many times letter i follows j
    n-grams
    how many doubles/tripples/quadruplets

"""

x = np.loadtxt("best_f.txt")

print(x)
print(x.shape)

raise

def uniqueLength(row):
    return len(list(set(row['f_27'])))

def preprocess(df) -> pd.DataFrame:

    # Categorize the string
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    df['f_27_num'] = df.apply(uniqueLength, axis=1) # nr. of unique chars in string
    #le = LabelEncoder()
    #df['f_27_num'] = le.fit_transform(df['f_27_num'])
    #df = df.drop(columns=['f_27'])
    df = df.drop(columns=['id'])
    df = df.drop(columns=[f'f_{i:02}' for i in range(27)])
    df = df.drop(columns=['f_28', 'f_29', 'f_30'])
    return df


def sum_string(row):
    return sum([ord(s)-ord('A') for s in list(row.f_27)])
def longest_subseq(row):
    groups = groupby(row['f_27'])
    g = [(label, sum(1 for _ in group)) for label, group in groups]
    m = max(g, key=itemgetter(1))[1]
    return m
def longest_subseq_elem(row):
    groups = groupby(row['f_27'])
    g = [(label, sum(1 for _ in group)) for label, group in groups]
    (elem, val) = max(g, key=itemgetter(1))
    if val == 1:
        return "equal"
    else:
        return elem
def first_element(row):
    return list(row.f_27)[0]
def last_element(row):
    return list(row.f_27)[-1]

"""
X = pd.read_csv("Data/train.csv")
X = preprocess(X)
strings = list(X['f_27'].values)
print(X.head())
X['string_sum'] = X.apply(sum_string, axis=1)
X['first_elem'] = X.apply(first_element, axis=1)
le = LabelEncoder()
X['first_elem'] = le.fit_transform(X['first_elem'])
X['last_elem'] = X.apply(last_element, axis=1)
le = LabelEncoder()
X['last_elem'] = le.fit_transform(X['last_elem'])
X['longest_seq'] = X.apply(longest_subseq, axis=1) # nr. of unique chars in string
X['longest_seq_elem'] = X.apply(longest_subseq_elem, axis=1)
le = LabelEncoder()
X['longest_seq_elem'] = le.fit_transform(X['longest_seq_elem'])
print(X.head())
"""

train = pd.read_parquet("Data/pre_processed_train.parquet")
#test = pd.read_parquet("Data/pre_processed_test.parquet")

print(train.iloc[0])

