from sklearn.ensemble import RandomForestClassifier

def random_forest_model(X_train, X_test, y_train):

    clf = RandomForestClassifier(max_depth= 9, max_features= sqrt, min_samples_leaf= 3, min_samples_split= 9, n_estimators= 194)

    # fit the classifier to the training data
    clf.fit(X_train, y_train)

    # make predictions on the test set
    return clf.predict(X_test)