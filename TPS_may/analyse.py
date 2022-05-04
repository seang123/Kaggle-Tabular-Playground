import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, auc, roc_curve
from xgboost import XGBClassifier
import reduce_mem
import xgboost as xgb
import xgbfir
import time
import sys

"""
    Dataframe chaining - tomaugspurger.github.io/method-chaining.html
"""

## Parameters
home_dir = '/home/hpcgies1/Projects/TPS_may/'

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        print(f"> {func.__name__} - {(time.perf_counter() - start):.3f} sec")
        return out
    return wrapper

def read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def uniqueLength(row):
    return len(list(set(row['f_27'])))

def preprocess(df) -> pd.DataFrame:

    # Categorize the string
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    df['f_27_num'] = df.apply(uniqueLength, axis=1) # nr. of unique chars in string
    #le = LabelEncoder()
    #df['f_27_num'] = le.fit_transform(df['f_27_num'])
    df = df.drop(columns=['f_27'])
    df = df.drop(columns=['id'])
    return df

def z_score(X, Y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.transform(Y)
    return X, Y

def boost_model(X_train, X_test, y_train, y_test):
    """ XGB model - reaches 96.6% AUC after 700 iterations """
    params = {'n_estimators': 2000, #900, # 4096, # nr. boosting rounds
            'max_depth': 12, # 8,
            'max_leaves': 0,
            'learning_rate': 0.15,
            'subsample': 0.95,
            'colsample_bytree': 0.95,
            'reg_alpha': 1.5, # L1
            'reg_lambda': 1.5, # L2
            'gamma': 1.5,
            'booster': 'gbtree', #'dart', #'gbtree', # gbtree - default
            'random_state': 46,
            #'scale_pos_weight':0, # 1 -defualt for when high class imbalance
            'objective': 'binary:logistic',
            'base_score': 0.49, # initial prediction score of all instances (global bias)
            'tree_method': 'hist', # 'approx', # 'gpu_hist' for gpu training
            'early_stopping_rounds':256,
            'eval_metric':['auc'],
            }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose=50)
    print(model.feature_importances_)

    """ # Save model
    bst = model.get_booster()
    bst.feature_names = list(feature_names)
    bst.dump_model('xgb.dump', with_stats=True)
    xgbfir.saveXgbFI(bst, feature_names = feature_names)
    """

    ypreds = model.predict(X_test) # (n, 2)
    print("ROC-AUC:", roc_auc_score(y_test, ypreds))

    ypredsp = model.predict_proba(X_test)[:,1]
    np.save(open(f"xgb_predict.npy", "wb"), ypredsp)

    return model

def grid_search(train, target):
    """ Grid search over XGBoost parameters
    """
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1001)
    param_grid = {
            'n_estimators': [50], #900, # 4096, # nr. boosting rounds
            'max_depth': [5, 8, 12], # 8,
            'max_leaves': [0, 2, 4],
            'learning_rate': [0.15],
            'grow_policy': [0, 1],
            'subsample': [0.95],
            'colsample_bytree': [0.95],
            'reg_alpha': [1.0, 1.5, 2.],  # L1
            'reg_lambda': [1.0, 1.5, 2.], # L2
            'gamma': [0.5, 1, 1.5, 2],
            'booster':['gbtree'],
            'random_state': [46],
            'objective': ['binary:logistic'],
            'base_score': [0.5],
            'tree_method': ['hist'],
            #'early_stopping_rounds':[256],
            'eval_metric':['auc'],
        }
    model = XGBClassifier()
    grid_cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=skf.split(train,target), scoring="roc_auc")
    _ = grid_cv.fit(train, target)
    print("best param:\n", grid_cv.best_params_)
    print("best score: ", grid_cv.best_score_)


def boost_model_(X_train, X_test, y_train, y_test, param):
    """ Fit model - eval on val set """
    model = XGBClassifier(**param).fit(
            X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    ypreds = model.predict(X_test)
    print("ROC-AUC:", roc_auc_score(y_test, ypreds))
    return model

def mlp(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier

    m = MLPClassifier(
            #hidden_layer_sizes=(64, 64, 16, 1),
            hidden_layer_sizes=(256, 128, 64, 16, 1),
            activation = 'relu',
            solver='adam',
            alpha=0.0001,
            batch_size=4096,
            learning_rate_init=0.0001,
            learning_rate='constant', # 'adaptive', #'constant',
            max_iter = 600,
            random_state=46,
            verbose=True,
            momentum=0.9,
            validation_fraction=0.1,
    ).fit(X_train, y_train)

    print("score:", m.score(X_test, y_test))
    ypred = m.predict(X_test)
    print("ROC-AUC:", roc_auc_score(y_test, ypred))
    #fpr, tpr, threshold = roc_curve(y_test, ypred)

    #ypredp = m.predict_proba(X_test)[:,1]
    #np.save(open(f"mlp_predict.npy", "wb"), ypredp)
    return m

def generate_submission(pred, name: str = 'submission'):
    #x = list(zip(tid, pred))
    #sub = pd.DataFrame.from_records(x, columns=['id', 'target'])
    #sub.to_csv(f"{home_dir}/submission.csv", index=False)
    sub = pd.read_csv(f"{home_dir}/Data/sample_submission.csv")
    sub['target'] = pred
    sub.to_csv(f"{home_dir}/{name}.csv", index=False)
    print("-- Submission generated --")

def poly_builder(df_train, df_test, degree=2):
    cols = ['f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']
    pb = PolynomialFeatures(degree=degree, include_bias=False)

    df_poly = pb.fit_transform(df_train[cols])
    df_poly_test = pb.transform(df_test[cols])
    col_names = [f'poly_{i}' for i in range(df_poly.shape[1])]
    df_poly = pd.DataFrame(df_poly, columns=col_names)
    df_poly_test = pd.DataFrame(df_poly_test, columns=col_names)
    df = pd.concat([df_train, df_poly], axis=1)
    df_test = pd.concat([df_test, df_poly_test], axis=1)
    return df, df_test

def poly_builder_mf(df_train, df_test, degree=2):
    """ Generate polynomial features """
    # Features which might have a higher-order interaction
    cols = [['f_00', 'f_01', 'f_02', 'f_05'], ['f_21', 'f_22'], ['f_20', 'f_25'], ['f_23', 'f_28'], ['f_28', 'f_25', 'f_23', 'f_20'], ['f_19', 'f_24'], ['f_03', 'f_04', 'f_06']]
    #cols = [['f_26'], ['f_25'], ['f_28'], ['f_24'], ['f_20'], ['f_21'], ['f_22'], ['f_23'], ['f_19']]
    cols = [['f_03', 'f_04', 'f_06'], ['f_28', 'f_25'], ['f_23', 'f_20']]
    #cols = ['f_26', 'f_21', 'f_30', 'f_27_num', 'f_22', 'f_01', 'f_02', 'f_00', 'ch1', 'ch6', 'f_05', 'f_24', 'f_19', 'f_25', 'f_28', 'f_29']

    pb = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    for i, c in enumerate(cols):
        poly = pb.fit_transform(df_train[c])
        poly_test = pb.transform(df_test[c])
        poly = poly[:,len(c):] # remove the duplicate features
        poly_test = poly_test[:,len(c):]
        col_names = [f'poly_{i}_{x}' for x in range(poly.shape[1])]
        poly = pd.DataFrame(poly, columns=col_names)
        poly_test = pd.DataFrame(poly_test, columns=col_names)
        df_train = pd.concat([df_train, poly], axis=1)
        df_test  = pd.concat([df_test, poly_test], axis=1)
    """
    poly = pb.fit_transform(df_train[cols])
    poly_test = pb.transform(df_test[cols])
    poly = poly[:,len(cols):] # remove the duplicate features
    poly_test = poly_test[:,len(cols):]
    col_names = [f'poly_{x}' for x in range(poly.shape[1])]
    poly = pd.DataFrame(poly, columns=col_names)
    poly_test = pd.DataFrame(poly_test, columns=col_names)
    df_train = pd.concat([df_train, poly], axis=1)
    df_test  = pd.concat([df_test, poly_test], axis=1)
    df_train.drop(columns=cols)
    df_test.drop(columns=cols)
    """
    return df_train, df_test


def main():

    X_train, X_test, y_train, y_test, train, test = load_data()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    """
    model = mlp(X_train, X_test, y_train, y_test)
    ypreds = model.predict_proba(test)[:,1]
    generate_submission(ypreds, name='mlp_submission_poly_d3')
    """

    model = boost_model(X_train, X_test, y_train, y_test)
    ypreds = model.predict_proba(test)[:,1] # (n, 2)
    generate_submission(ypreds, name='xgb_submission_poly')

@timeit
def load_data():
    """
    s = time.time()
    train = preprocess(read(f"{home_dir}/train.csv")) # (900_000, 42)
    print(f"load + preprocess: {(time.time() - s):.4f}")
    test = preprocess(read(f"{home_dir}/test.csv")) # (700_000, 42)
    print("train:", train.shape)
    print("test: ", test.shape)
    train = reduce_mem.reduce_memory_usage(train)
    test  = reduce_mem.reduce_memory_usage(test)
    train.to_csv(f"{home_dir}/train_mod.csv")
    test.to_csv(f"{home_dir}/test_mod.csv")
    """
    s = time.time()
    train = pd.read_csv(f"{home_dir}/Data/train_mod.csv")
    test = pd.read_csv(f"{home_dir}/Data/test_mod.csv")
    print(f"load: {(time.time() - s):.4f}")

    # Polynomials
    train, test = poly_builder_mf(train, test, degree=4)

    target = train['target'].values
    train.drop(columns=['target'], inplace=True)
    feature_names = train.columns
    train, test = z_score(train, test)
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.005, random_state=4242)

    return X_train, X_test, y_train, y_test, train, test

if __name__ == '__main__':
    main()
