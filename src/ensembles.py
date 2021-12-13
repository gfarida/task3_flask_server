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
        self.rmse_train = []
        self.times = []
        self.rmse_val = []

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
        rmse_val_bool = False
        if (X_val is not None) and (y_val is not None):
            rmse_val_bool = True
            y_pred_val = 0

        y_pred_train = 0
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
            if rmse_val_bool:
                y_pred_val += regressor.predict(X_val[:, feat_idx])
                self.rmse_val.append(np.sqrt(((y_pred_val / (i + 1) - y_val) ** 2).sum() / X_val.shape[0]))

            y_pred_train += regressor.predict(x_train)
            self.rmse_train.append(np.sqrt(((y_pred_train / (i + 1) - y) ** 2).sum() / x_train.shape[0]))

            cur_time += time() - start
            self.times.append(cur_time)

        if rmse_val_bool:
            return self.rmse_train, self.rmse_val, self.times
        else:
            return self.rmse_train, self.times

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
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_s_s = feature_subsample_size
        self.trees_param = trees_parameters
        self.models = []
        self.features = []
        self.alphas = []
        self.rmse_train = []
        self.times = []
        self.rmse_val = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_s_s is None:
            self.feature_s_s = X.shape[1] // 3

        f = 0
        f_val = 0

        rmse_val_bool = False
        if (X_val is not None) and (y_val is not None):
            rmse_val_bool = True

        cur_time = 0

        def loss(alpha, y, f, y_pred):
            return ((f + alpha * y_pred - y) ** 2).sum()

        for i in range(self.n_estimators):
            start = time()
            feat_idx = np.random.choice(X.shape[1], self.feature_s_s, replace=False)
            self.features.append(feat_idx)
            regressor = DecisionTreeRegressor(max_depth=self.max_depth, **self.trees_param)
            self.models.append(regressor)
            x_train = X[:, feat_idx]
            regressor.fit(x_train, y - f)
            y_pred = regressor.predict(x_train)
            alpha = minimize_scalar(loss, args=(y, f, y_pred)).x
            self.alphas.append(alpha)
            if rmse_val_bool:
                y_pred_val = regressor.predict(X_val[:, feat_idx])
                f_val += alpha * self.learning_rate * y_pred_val
                self.rmse_val.append(np.sqrt(((f_val - y_val) ** 2).sum() / X_val.shape[0]))

            f += alpha * self.learning_rate * y_pred
            self.rmse_train.append(np.sqrt(((f - y) ** 2).sum() / x_train.shape[0]))

            cur_time += time() - start
            self.times.append(cur_time)

        if rmse_val_bool:
            return self.rmse_train, self.rmse_val, self.times
        else:
            return self.rmse_train, self.times

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = 0
        i = 0
        for model in self.models:
            y_pred = model.predict(X[:, self.features[i]])
            res += y_pred * self.alphas[i] * self.learning_rate
            i += 1
        return res
