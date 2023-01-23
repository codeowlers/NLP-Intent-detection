import pandas as pd

def array_column_spread(df, column_name):
    df[['{}_{}'.format(column_name, i) for i in range(len(df[column_name].iloc[0]))]] = df[column_name].apply(lambda x: pd.Series(x))
    df.drop(columns=[column_name], inplace=True)