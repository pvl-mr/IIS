from lin_reg import LinReg
from pol_reg import PolReg
from perceptron import Pers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

X, y = make_circles(noise=0.2, factor=0.5, random_state=None)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

lin_reg = LinReg(X, y, X_train, X_test, y_train, y_test)
pol_reg = PolReg(X, y, X_train, X_test, y_train, y_test)
pers = Pers(X, y, X_train, X_test, y_train, y_test)
models = [lin_reg, pol_reg, pers]

model = lin_reg.create_model()
evaluation = lin_reg.evaluate_model(model)
plot0 = lin_reg.make_plot(model, evaluation)


model = pol_reg.create_model()
evaluation = pol_reg.evaluate_model(model)
plot1 = pol_reg.make_plot(model, evaluation)


model = pers.create_model()
evaluation = pers.evaluate_model(model)
plot2 = pers.make_plot(model, evaluation)


