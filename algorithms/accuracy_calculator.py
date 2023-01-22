from sklearn import metrics


def accuracy_calculator(y_test, y_pred):
    # returning the accuracy
    return metrics.accuracy_score(y_test, y_pred)