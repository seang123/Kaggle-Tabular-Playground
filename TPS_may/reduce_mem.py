import pandas as pd
import numpy as np

def reduce_memory_usage(df, verbose=True):

    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.in64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print(f"Mem. usage decreased to {end_mem:.2f} Mb ({(100 * (start_mem - end_mem) / start_mem):.1f}% reduction)")
    return df



if __name__ == '__main__':
    df = pd.read_csv("/home/hpcgies1/Projects/TPS_may/train_mod.csv")
    df2 = pd.read_csv("/home/hpcgies1/Projects/TPS_may/test_mod.csv")
    train = reduce_memory_usage(df)
    test = reduce_memory_usage(df2)

    train.to_csv("/home/hpcgies1/Projects/TPS_may/train_mod.csv")
    test.to_csv("/home/hpcgies1/Projects/TPS_may/test_mod.csv")

