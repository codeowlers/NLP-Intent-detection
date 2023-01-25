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
        label_encoder(df,column_name)


def one_hot_encoder(df, column):
    
    encoder = OneHotEncoder()
     # Fit and transform the data
    encoded_data = encoder.fit_transform(df[[column]])

    # Get the feature names
    feature_names = encoder.get_feature_names_out([column])

    # Create a new DataFrame with the encoded data
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=feature_names)

    # Drop the original column
    df.drop(column, axis=1, inplace=True)

    #update the dataframe with the encoded dataframe
    df = pd.concat([df, encoded_df], axis=1)
    print(df)


def one_hot_encode_columns(df,columns):
    for column in columns:
        one_hot_encoder(df, column)