
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from itertools import groupby
from operator import itemgetter


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"> {func.__name__} - {(time.perf_counter() - start):.3f} sec")
        return out
    return wrapper

def uniqueLength(row):
    return len(list(set(row['f_27'])))

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

@timeit
def pre_process(df):
    """ Pre-process the dataframe """
    df = df.drop(columns=['id'])
    # Categorize the string
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    df['f_27_num'] = df.apply(uniqueLength, axis=1) # nr. of unique chars in string

    ## Additional encoding of information in f_27
    df['string_sum'] = df.apply(sum_string, axis=1)
    df['first_elem'] = df.apply(first_element, axis=1)
    le = LabelEncoder()
    df['first_elem'] = le.fit_transform(df['first_elem'])
    df['last_elem'] = df.apply(last_element, axis=1)
    le = LabelEncoder()
    df['last_elem'] = le.fit_transform(df['last_elem'])
    df['longest_seq'] = df.apply(longest_subseq, axis=1) # nr. of unique chars in string
    df['longest_seq_elem'] = df.apply(longest_subseq_elem, axis=1)
    le = LabelEncoder()
    df['longest_seq_elem'] = le.fit_transform(df['longest_seq_elem'])

    df = df.drop(columns=['f_27'])

    return df

def z_score(X: pd.DataFrame, Y: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.transform(Y)
    return X, Y

@timeit
def load_data():
    train_df = pd.read_csv("Data/train.csv")
    test_df  = pd.read_csv("Data/test.csv")

    train_df_shape = train_df.shape
    test_df_shape = test_df.shape

    # Extract target info
    target = train_df['target'].values
    train_df = train_df.drop(columns=['target'])

    # Temporarily combine train-test for preprocessing
    df = train_df.append(test_df, ignore_index=True)

    # Preprocess
    df = pre_process(df)

    # Split
    train_df = df.iloc[:train_df_shape[0]]
    test_df = df.iloc[train_df_shape[0]:]

    print("train_df:", type(train_df))
    print("test_df:", type(test_df))

    assert train_df.shape[0] == train_df_shape[0], f"{train_df.shape} != {train_df_shape[0]}"
    assert test_df.shape[0] == test_df_shape[0], f"{test_df.shape} != {test_df_shape[0]}"

    # Z-score
    train, test = z_score(train_df, test_df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.005, random_state=4242)

    return X_train, X_test, y_train, y_test, train, test
