from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def label_encoder(df, column_name):
    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    label = le.fit_transform(df[column_name])
    # removing the column 'Purchased' from df
    # as it is of no use now.
    df.drop(column_name, axis=1, inplace=True)

    # Appending the array to our dataFrame
    # with column name 'Purchased'
    df[column_name] = label


def label_encode_columns(df, columns_names):
    for column_name in columns_names:
        label_encoder(df, column_name)


def one_hot_encoder(df, column, n_minus_1=True):
    encoded_df = pd.get_dummies(df[column], drop_first=n_minus_1)
    df = df.drop(columns=[column], axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def one_hot_encode_columns(df, columns, n_minus_1=True):
    for column in columns:
        df = one_hot_encoder(df, column, n_minus_1)
    return df
