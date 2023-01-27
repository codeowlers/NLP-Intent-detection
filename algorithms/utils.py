import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def array_column_spread(df, column_name):
    df[['{}_{}'.format(column_name, i) for i in range(len(df[column_name].iloc[0]))]] = df[column_name].apply(lambda x: pd.Series(x))
    df.drop(columns=[column_name], inplace=True)



def time_domain_2D(df, column):
    df[f'{column}_mean'] = df[column].apply(lambda x: np.mean(x,axis=1))
    df[f'{column}_min'] = df[column].apply(lambda x: np.min(x,axis=1))
    df[f'{column}_max'] = df[column].apply(lambda x: np.max(x,axis=1))
    df[f'{column}_skew'] = df[column].apply(lambda x: skew(x,axis=1))
    df[f'{column}_kurtosis'] = df[column].apply(lambda x: kurtosis(x,axis=1))
    df[f'{column}_std'] = df[column].apply(lambda x: np.std(x,axis=1))


def time_domain_1D(df, column):
    df[f'{column}_mean'] = df[column].apply(lambda x: np.mean(x))
    df[f'{column}_min'] = df[column].apply(lambda x: np.min(x))
    df[f'{column}_max'] = df[column].apply(lambda x: np.max(x))
    df[f'{column}_skew'] = df[column].apply(lambda x: skew(x))
    df[f'{column}_kurtosis'] = df[column].apply(lambda x: kurtosis(x))
    df[f'{column}_std'] = df[column].apply(lambda x: np.std(x))