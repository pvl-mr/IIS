import numpy as np
from sklearn.linear_model import LinearRegression
from plotting_model import PlottingModel


class LinReg(PlottingModel):

    def __init__(self, X, y, X_train, X_test, y_train, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def create_model(self):
        lin_reg = LinearRegression()
        lin_reg.fit(self.X_train, self.y_train)
        return lin_reg

    def evaluate_model(self, model):
        from sklearn.metrics import mean_squared_error
        y_predict = model.predict(self.X_test)
        lin_mse = mean_squared_error(self.y_test, y_predict)
        lin_rmse = np.sqrt(lin_mse)
        return lin_rmse

