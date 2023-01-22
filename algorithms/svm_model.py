from sklearn.svm import SVC


def svm_model(X_train, y_train, X_test):
    clf = SVC()
    # train the model on the training data
    clf.fit(X_train, y_train)
    # predict the target values for the test data
    # returning the y_predict
    return clf.predict(X_test)