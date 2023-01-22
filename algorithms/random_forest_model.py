from sklearn.ensemble import RandomForestClassifier

def random_forest_model(X_train, X_test, y_train):

    clf = RandomForestClassifier(n_estimators=100, random_state=0)

    # fit the classifier to the training data
    clf.fit(X_train, y_train)

    # make predictions on the test set
    return clf.predict(X_test)