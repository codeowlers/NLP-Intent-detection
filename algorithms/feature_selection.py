from sklearn.svm import SVC
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def select_top_n_features(X, y):
    # initialize the RF model
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    # initialize Boruta
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    # fit Boruta
    feat_selector.fit(X.values, y)
    # get the selected feature mask
    mask = feat_selector.support_
    # apply the mask to the dataframe to get the top n features
    top_n_features = X.columns[mask]
    return top_n_features


def select_top_n_features_with_grid_search(dataframe, target, n_features, param_grid={'kernel': ['linear', 'rbf'],
              'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 'auto'],
              'class_weight': ['balanced', None]}, cv = 5):
    X = dataframe.drop(target, axis=1)
    y = dataframe[target]
    # initialize the SVC model with the param_grid
    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    svc = grid_search.best_estimator_
    # initialize Boruta
    feat_selector = BorutaPy(svc, n_estimators='auto', verbose=2, random_state=1)
    # fit Boruta
    feat_selector.fit(X.values, y.values)
    # get the selected feature mask
    mask = feat_selector.support_
    # apply the mask to the dataframe to get the top n features
    top_n_features = X.columns[mask]
    return top_n_features,grid_search.best_params_,grid_search.best_score_

