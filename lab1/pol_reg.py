import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from plotting_model import PlottingModel


class PolReg(PlottingModel):
    def __init__(self, X, y, X_train, X_test, y_train, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_model(self):
        my_polynomial_model = PolynomialFeatures(degree=4, include_bias=False)
        lin = linear_model.LinearRegression()
        pipeline = Pipeline(
            [("polynomial_features", my_polynomial_model), ("linear_regression", lin)])
        return pipeline

# Don't working render
# poly = PolynomialFeatures(degree=4)
#
# X_train = poly.fit_transform(X_train)
# X_test = poly.fit_transform(X_test)
#
# pol_reg = linear_model.LinearRegression()
# pol_reg.fit(X_train, y_train)

# print(pol_reg.score(X_train, y_train))

    def evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        pol_scores = cross_val_score(model, self.X_test, self.y_test,
                                     scoring="neg_mean_squared_error", cv=10)
        pol_rmse_scores = np.sqrt(-pol_scores).mean()
        return pol_rmse_scores


