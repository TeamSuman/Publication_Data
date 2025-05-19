#from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings

# comment out the next line to see the warning
warnings.simplefilter('ignore', category=ConvergenceWarning)
import numpy as np #type: ignore
from sklearn.svm import SVR #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor #type: ignore
from sklearn.gaussian_process.kernels import RBF #type: ignore
from sklearn.kernel_ridge import KernelRidge #type: ignore
from sklearn.model_selection import GridSearchCV #type: ignore
from sklearn.linear_model import LinearRegression #type: ignore
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel #type: ignore
from xgboost import XGBRegressor #type: ignore

class Regressor:
    def __init__(self):
        pass

    def svr(self, X, y):
        hyperparameters = {'C': np.linspace(0.01,10, 20), 'gamma': np.linspace(0.01,2, 20),
                        'kernel': ['rbf']}
        model = GridSearchCV(SVR(),hyperparameters, n_jobs = -1)
        model.fit(X, y)
        return model

    #@ignore_warnings(category=ConvergenceWarning)
    def gpr(self, X, y):
        param_grid = {
            "alpha":  [1e-2, 1e-3],
            "kernel": [RBF(l) for l in np.logspace(-6, 2, 10)]
        }

        gp = GaussianProcessRegressor()

        clf = GridSearchCV(estimator=gp, param_grid=param_grid, cv=5, n_jobs = -1)
        clf.fit(X, y)
        return clf
    def krr(self, x, y):

        # Define kernel function
        kernel = 'rbf'

        # Define hyperparameters to optimize
        hyperparameters = {'alpha': np.linspace(0.01,2, 20), 'gamma': np.linspace(0.01,2, 20)}

        # Train kernel ridge regression model with cross-validation
        model = GridSearchCV(KernelRidge(kernel=kernel), hyperparameters, cv=5, n_jobs = -1)
        model.fit(x, y)

        return model

    def lr(self, x, y):

        model = LinearRegression()
        model.fit(x, y)

        return model

    def rfr(self, x, y):

        param_grid = {
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rfr = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, scoring='neg_mean_squared_error')

        grid_search.fit(x, y)
        #best_model = grid_search.best_estimator_

        return grid_search

    def xgbr(self, X, y):
        # Create an XGBoost regressor object
        xgbr = XGBRegressor()

        # Define the hyperparameters to be tuned
        param_grid = {'n_estimators': [100, 500, 1000],
                    'max_depth': [3, 5, 7],}

        # Perform grid search to find the best hyperparameters
        grid_search = GridSearchCV(estimator=xgbr, param_grid=param_grid, cv=5, n_jobs=1)
        grid_search.fit(X, y)

        return grid_search
