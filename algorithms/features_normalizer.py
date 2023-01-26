from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(df):
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df