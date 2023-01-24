from sklearn.svm import SVC

# [(0.1, 'linear', 0.1), (0.1, 'linear', 1), (0.1, 'poly', 0.1), (0.1, 'poly', 1), (0.1, 'rbf', 0.1), (0.1, 'rbf', 1), (1, 'linear', 0.1), (1, 'linear', 1), (1, 'poly', 0.1), (1, 'poly', 1), (1, 'rbf', 0.1), (1, 'rbf', 1), (10, 'linear', 0.1), (10, 'linear', 1), (10, 'poly', 0.1), (10, 'poly', 1), (10, 'rbf', 0.1), (10, 'rbf', 1)]

def svm_model(X_train, y_train, X_test):
    clf = SVC(C=0.1, kernel='linear', gamma=0.1)
    # train the model on the training data
    clf.fit(X_train, y_train)
    # predict the target values for the test data
    # returning the y_predict
    return clf.predict(X_test)