import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, auc, roc_curve, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import reduce_mem
import xgboost as xgb
import xgbfir
import time
import sys
import pipeline
import pipeline2

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

    params = {'n_estimators': 1500, #900, # 4096, # nr. boosting rounds
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
            'early_stopping_rounds':1000,
            'eval_metric':['auc'],
            }
    model = XGBClassifier(**params)

    #X_train = np.concatenate((X_train, X_test), axis=0)
    #y_train = np.concatenate((y_train, y_test), axis=0)
    #print("concat X_train:", X_train.shape)

    model.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose=50)
    #print(model.feature_importances_)

    """ # Save model
    bst = model.get_booster()
    bst.feature_names = list(feature_names)
    bst.dump_model('xgb.dump', with_stats=True)
    xgbfir.saveXgbFI(bst, feature_names = feature_names)
    """

    display_score(model, X_train, y_train, 'Train')
    display_score(model, X_test, y_test, 'Val')

    return model

def grid_search(train, target):
    """ Grid search over XGBoost parameters
    """
    #skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1001)
    param_grid = {
            'n_estimators': [200], #900, # 4096, # nr. boosting rounds
            'max_depth': [5, 8, 12], # 8,
            'max_leaves': [2, 4],
            'learning_rate': [0.15],
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
            'eval_metric':['auc'],
        }
    model = XGBClassifier()
    grid_cv = GridSearchCV(model, param_grid, n_jobs=-1, scoring="roc_auc")#, cv=skf.split(train,target), scoring="roc_auc")
    _ = grid_cv.fit(train, target)
    print("best param:\n", grid_cv.best_params_)
    print("best score: ", grid_cv.best_score_)

@timeit
def dtree(X_train, X_test, y_train, y_test, **kwargs):
    """ baseline: 1.0 train .77 validation score """
    from sklearn.tree import DecisionTreeClassifier, export_text, ExtraTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    m = DecisionTreeClassifier(max_features=None, max_depth=16) # 16 is optimal
    #m = ExtraTreesClassifier(n_estimators=200, n_jobs=-1)
    #m = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1)
    m = m.fit(X_train, y_train)

    #r = export_text(m, feature_names=list(kwargs.get('feature_names')))
    #print(r)
    display_score(m, X_train, y_train, 'Train')
    display_score(m, X_test, y_test, 'Val')
    return m

def display_score(model, data, labels, name='Val'):
    ypreds_p = model.predict_proba(data)[:,1]
    ypreds   = model.predict(data)
    print(f"{name}:")
    print(f"\taccuracy: {accuracy_score(labels, ypreds):.4f}")
    print(f"\troc-auc:  {roc_auc_score(labels, ypreds_p):.4f}")
    print(f"\tF1:       {f1_score(labels, ypreds, average='macro'):.4f}")

def mlp(X_train, X_test, y_train, y_test):

    m = MLPClassifier(
            hidden_layer_sizes=(64, 64, 16, 1),
            activation = 'relu',
            solver='sgd',#'adam',
            #alpha=0.0001,
            alpha=0.32,
            batch_size=4096,
            learning_rate_init=0.0008,#0.0001,
            learning_rate='constant', # 'adaptive', #'constant',
            max_iter = 1200,
            random_state=46,
            verbose=True,
            momentum=0.9,
            validation_fraction=0.05,
    ).fit(X_train, y_train)

    display_score(m, X_train, y_train, 'Train')
    display_score(m, X_test, y_test, 'Val')
    return m

def generate_submission(pred, name: str = 'submission'):
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
    cols = [['f_03', 'f_04', 'f_06'], ['f_28', 'f_25'], ['f_23', 'f_20']]

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
    return df_train, df_test

def k_fold_cv(train, targets):
    """ KFold cross validation """
    kf = KFold(n_splits=5)
    for train_idx, test_idx in kf.split(train):
        X_train, X_test = train[train_idx], train[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

    model = boost_model(X_train, X_test, y_train, y_test)
    return model

def monte_carlo_cv(train, targets):
    from sklearn.model_selection import ShuffleSplit

    kf = ShuffleSplit(test_size=0.1, train_size=0.8, n_splits=10)
    for train_idx, test_idx in kf.split(train):
        X_train, X_test = train[train_idx], train[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]

        model = boost_model(X_train, X_test, y_train, y_test)
    return

def rec_feat_select(X_train, y_train):

    from sklearn.feature_selection import RFE
    dtree = DecisionTreeClassifier()
    selector = RFE(dtree, n_features_to_select=30, step=1)
    selector = selector.fit(X_train, y_train)
    print(selector.support_)
    print(selector.ranking_)


def main():

    X_train, X_test, y_train, y_test, targets, feature_names, train, test = pipeline.load_data(load_cache=True)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #model = dtree(X_train, X_test, y_train, y_test, feature_names = feature_names)
    #ypreds = model.predict_proba(test)[:,1]
    #generate_submission(ypreds, name='dtree_submission')
    #sys.exit(0)

    model = mlp(X_train, X_test, y_train, y_test)
    ypreds = model.predict_proba(test)[:,1]
    generate_submission(ypreds, name='mlp_submission_sgd')
    sys.exit(0)

    #grid_search(X_train, y_train)

    #monte_carlo_cv(train, targets)

    model = boost_model(X_train, X_test, y_train, y_test)
    ypreds = model.predict_proba(test)[:,1] # (n, 2)
    print("ypreds:", ypreds.shape)
    #np.save(open("xbg_predictions_test.npy", "wb"), ypreds)
    generate_submission(ypreds, name='xgb_submission')


if __name__ == '__main__':
    main()
