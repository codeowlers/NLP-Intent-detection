from sklearn.preprocessing import LabelEncoder

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


def encode_columns(df, columns_names):
    for column_name in columns_names:
        label_encoder(df,column_name)