from sklearn.model_selection import KFold
import numpy as np

def balance_df_kfold(df):
    kf = KFold(n_splits=5, shuffle=True)
    clean_idx = []
    noise_idx = []
    for label_col in df.select_dtypes(include=['object', 'int', 'float']).columns:
        for train_index, test_index in kf.split(df):
            train_data = df.iloc[train_index]
            test_data = df.iloc[test_index]
            # extract data and labels
            data = train_data.drop(label_col, axis=1)
            labels = train_data[label_col]

            for i in range(len(data)):
                # calculate the number of unique labels for the k nearest neighbors
                k = 5
                k_neighbors = np.argsort(np.linalg.norm(data.values - data.values[i], axis=1))[:k]
                unique_labels = len(set([labels.values[j] for j in k_neighbors]))

                # if there are more than 2 unique labels among the k nearest neighbors,
                # add the index to the clean data
                if unique_labels > 2:
                    clean_idx.append(i)
                else:
                    noise_idx.append(i)
    # return the cleaned dataframe
    return df.iloc[clean_idx],noise_idx