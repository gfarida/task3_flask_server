import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from time import time


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_s_s = feature_subsample_size
        self.trees_param = trees_parameters
        self.models = []
        self.features = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        rmse_val = False
        if not(X_val is None) and not(y_val is None):
            rmse_val = True
            loss_val = []
            y_pred_val = 0

        loss_train = []
        y_pred_train = 0
        times = []
        cur_time = 0

        if self.feature_s_s is None:
            self.feature_s_s = X.shape[1] // 3

        for i in range(self.n_estimators):
            start = time()
            obj_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            feat_idx = np.random.choice(X.shape[1], self.feature_s_s, replace=False)
            self.features.append(feat_idx)
            x_train = X[obj_idx, :][:, feat_idx]
            y_train = y[obj_idx]
            regressor = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_param)
            regressor.fit(x_train, y_train)
            self.models.append(regressor)
            if rmse_val:
                y_pred_val += regressor.predict(X_val[:, feat_idx])
                loss_val.append(np.sqrt((y_pred_val/(i + 1) - y) ** 2 / X_val.shape[0]))

            y_pred_train += regressor.predict(x_train)
            loss_train.append(np.sqrt((y_pred_train/(i + 1) - y) ** 2 / x_train.shape[0]))

            cur_time += time() - start
            times.append(cur_time)

        if rmse_val:
            return loss_train, loss_val, times
        else:
            return loss_train, times

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y_pred = 0
        i = 0
        for model in self.models:
            y_pred += model.predict(X[:, self.features[i]])
            i += 1
        return y_pred / self.n_estimators


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
