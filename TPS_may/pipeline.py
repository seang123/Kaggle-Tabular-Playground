
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from itertools import groupby
from operator import itemgetter
import reduce_mem


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"> {func.__name__} - {(time.perf_counter() - start):.3f} sec")
        return out
    return wrapper

def uniqueLength(row):
    return len(set(row['f_27']))

def sum_string(row):
    return sum([ord(s)-ord('A') for s in list(row.f_27)])

def longest_subseq_(row):
    groups = groupby(row['f_27'])
    g = [(label, sum(1 for _ in group)) for label, group in groups]
    (elem, val) = max(g, key=itemgetter(1))
    row['longest_seq'] = val
    row['longest_seq_elem'] = elem
    return row

def longest_subseq(row):
    groups = groupby(row['f_27'])
    g = [(label, sum(1 for _ in group)) for label, group in groups]
    (elem, val) = max(g, key=itemgetter(1))
    return val

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
    return list(row.f_27)[1]

class Timer(object):
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, type, value, traceback):
        print(f"{self.name} - {(time.time() - self.start):.3f}")

@timeit
def pre_process(df):
    """ Pre-process the dataframe """
    #df = df.drop(columns=['id'])
    # Categorize the string
    with Timer("str ord"):
        for i in range(10):
            df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    with Timer("unq chars"):
        df['f_27_num'] = df.apply(uniqueLength, axis=1) # nr. of unique chars in string

    df['non_unq_len'] = 10 - df['f_27_num']

    float_features = [f for f in df.columns if df[f].dtype =='float64' and f !='target']
    for f in float_features:
        df[f'inv_{f}'] = 1 / df[f]

    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int)
                    - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int)
                    - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int)
                    - (i_00_01_26 < -5.0).astype(int)



    ## slight correlation with target (0.1)
    df['string_sum'] = df.apply(sum_string, axis=1)
    """
    ## Slight correlation with target (-0.1)
    df['first_elem'] = df.apply(first_element, axis=1)
    df['last_elem'] = df.apply(last_element, axis=1)
    le = LabelEncoder()
    df['first_elem'] = le.fit_transform(df['first_elem'])
    le = LabelEncoder()
    df['last_elem'] = le.fit_transform(df['last_elem'])
    ## Doesn't correlate with the target
    df['longest_seq'] = df.apply(longest_subseq, axis=1) # nr. of unique chars in string
    df['longest_seq_elem'] = df.apply(longest_subseq_elem, axis=1)
    #df = df.apply(longest_subseq_, axis=1)
    le = LabelEncoder()
    df['longest_seq_elem'] = le.fit_transform(df['longest_seq_elem'])

    df['126'] = df['f_21'] + df['f_22'] + df['f_26']
    df['f_28'] = np.sqrt(abs(df['f_28']))
    df['f_25'] = np.sqrt(abs(df['f_25']))
    df['f_24'] = np.sqrt(abs(df['f_24']))
    df['f_04'] = np.sin(df['f_04']) - df['f_28']
    df['f_27_num_29_30'] = df['f_27_num'] + df['f_29'] + df['f_30']
    """

    #df = df.drop(columns=['f_27'])

    return df

def z_score(X: pd.DataFrame, Y: pd.DataFrame):
    """ Z-score each column """
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.transform(Y)
    return X, Y

def poly_features(df, df2, degree: int, features: list):
    """ Add polynomial features

    Parameters:
    -----------
        df - dataframe
        degree - polynomial degree
        features - features that should interact
    """
    pb = PolynomialFeatures(degree, include_bias=False)

    for i, c in enumerate(features):
        poly = pb.fit_transform(df[c])
        poly2 = pb.transform(df2[c])
        poly = poly[:,len(c):] # remove the duplicate features
        poly2 = poly2[:,len(c):]
        col_names = [f'poly_{i}_{x}' for x in range(poly.shape[1])]
        poly = pd.DataFrame(poly, columns=col_names)
        poly2 = pd.DataFrame(poly2, columns=col_names)
        df = pd.concat([df, poly], axis=1)
        df2 = pd.concat([df2, poly2], axis=1)

    return df, df2

def n_grams(ls: list, n: int = 2):
    grams = []
    for i in range(len(ls)):
        g = ls[i:i+n]
        grams.append(g)
    return [i for i in grams if len(i) == n]

def process_f_27(df):
    """ Process the f_27 string feature """
    le = LabelEncoder()
    f27 = df['f_27'].values
    f27 = np.array([list(i) for i in f27])
    #n_grams_2 = [n_grams(i, 2) for i in f27]
    f27_shape = f27.shape
    f27 = le.fit_transform(f27.reshape(-1))
    f27 = f27.reshape(f27_shape)

    return f27

def pseudo_labelling(train, test, targets):

    t = pd.read_csv("mlp_submission.csv")['target'].values
    t0 = np.where(t < 0.0001)[0]
    t1 = np.where(t > 0.9999)[0]

    print("Pre pseudo-labelling:", train.shape)
    train = np.concatenate([train, test[t0,:], test[t1,:]], axis=0)
    print("Post pseudo-labelling:", train.shape)

    labels = [i for i in t[t0]] + [i for i in t[t1]]
    labels = [0 if i <= 0.5 else 1 for i in labels]

    print("pre targets:", targets.shape)
    labels = np.concatenate([targets, labels], axis=0)
    print("post targets:", labels.shape)

    return train, test, labels



@timeit
def load_data(load_cache=False, save=False):

    if not load_cache:
        train_df = pd.read_csv("Data/train.csv")
        test_df  = pd.read_csv("Data/test.csv")
        train_df_shape = train_df.shape
        test_df_shape = test_df.shape

        # Temporarily combine train-test for preprocessing
        df = train_df.append(test_df, ignore_index=True)

        # Preprocess
        df = pre_process(df)

        # Encode f_27
        """
        f27 = process_f_27(df)
        f27_train = f27[:train_df_shape[0],:]
        f27_test  = f27[train_df_shape[0]:,:]
        """
        df = df.drop(columns=['f_27'])

        # Reduce memory footprint
        df = reduce_mem.reduce_memory_usage(df, verbose=True)

        # Split
        train_df = df.iloc[:train_df_shape[0]]
        test_df = df.iloc[train_df_shape[0]:]

        # Save to disk
        if save:
            train_df.to_parquet("Data/pre_processed_train.parquet", index=False)
            test_df.to_parquet("Data/pre_processed_test.parquet", index=False)

        print("train_df:", type(train_df))
        print("test_df:", type(test_df))

        assert train_df.shape[0] == train_df_shape[0], f"{train_df.shape} != {train_df_shape[0]}"
        assert test_df.shape[0] == test_df_shape[0], f"{test_df.shape} != {test_df_shape[0]}"
    else:
        train_df = pd.read_parquet("Data/pre_processed_train.parquet")
        test_df  = pd.read_parquet("Data/pre_processed_test.parquet")

    # Extract target
    cols_to_drop = ['target', 'id']#*[f'f_{i:02}' for i in range(0, 7)]]#, 'string_sum', 'longest_seq', 'longest_seq_elem', 'first_elem', 'last_elem']

    target = train_df['target'].values
    train  = train_df.drop(columns=cols_to_drop)
    test   = test_df.drop(columns=cols_to_drop)

    feature_names = train.columns
    print("train columns:", feature_names)

    # Poly features
    #train, test = poly_features(train, test, degree=3, features = [['f_22', 'f_26', 'f_21'], ['f_03', 'f_04', 'f_06'], ['f_28', 'f_04'], ['f_28', 'f_21', 'f_24']])

    # Z-score
    train, test = z_score(train, test)

    # Pseudo-labelling
    train, test, target = pseudo_labelling(train, test, target)

    # Split
    #X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(train, target, f27_train, test_size=0.1, random_state=4242) # 0.005
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.05, random_state=4242) # 0.005

    return X_train, X_test, y_train, y_test, target, feature_names, train, test

if __name__ == '__main__':

    _ = load_data(load_cache=False, save=False)
