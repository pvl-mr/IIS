from feature_ranking import FeatureRanking
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
Y = (10 * np.sin(np.pi * X[:, 0]) + 20 * (X[:, 2] - .5)) + np.random.normal(0, 1)
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))
data = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                                'x7', 'x8', 'x9', 'x10',
                                'x11', 'x12', 'x13', 'x14'])
data['y'] = Y

sns.pairplot(data, x_vars=['x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                           'x7', 'x8', 'x9', 'x10',
                           'x11', 'x12', 'x13', 'x14'], y_vars=['y'])
names = ["x%s" % i for i in range(1, 15)]

feature_ranking = FeatureRanking(X, Y)
lasso_score = feature_ranking.generate_lasso_model()
rnd_forest_reg_score = feature_ranking.generate_rnd_forest_reg()
f_reg_score = feature_ranking.generate_f_regressor()

ranks = dict()
ranks["Lasso"] = feature_ranking.rank_to_dict(lasso_score, names)
ranks["Random Forest Regressor"] = feature_ranking.rank_to_dict(rnd_forest_reg_score, names)
ranks["F-regressor"] = feature_ranking.rank_to_dict(f_reg_score, names)

print("Lasso", ranks["Lasso"])
print("Random Forest Regressor", ranks["Random Forest Regressor"])
print("F-regressor", ranks["F-regressor"])
print("Mean ", feature_ranking.calculate_mean(ranks))


plt.show()
