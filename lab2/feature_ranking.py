from operator import itemgetter

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


class FeatureRanking:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def generate_lasso_model(self):
        lasso = Lasso(alpha=.05)
        lasso.fit(self.X, self.Y)
        return lasso.coef_

    def generate_rnd_forest_reg(self):
        rnd_forest_reg = RandomForestRegressor()
        rnd_forest_reg.fit(self.X, self.Y)
        return rnd_forest_reg.feature_importances_

    def generate_f_regressor(self):
        scores, f_reg = f_regression(self.X, self.Y, center=True)
        return scores

    def rank_to_dict(self, coefs, names):
        coefs = np.abs(coefs)
        minmax = MinMaxScaler()
        coefs = minmax.fit_transform(np.array(coefs).reshape(14, 1)).ravel()
        coefs = map(lambda x: round(x, 2), coefs)
        return dict(zip(names, coefs))

    def calculate_mean(self, ranks):
        mean = {}
        for k, v in ranks.items():
            for item in v.items():
                if item[0] not in mean:
                    mean[item[0]] = 0
                mean[item[0]] += item[1]
        for k, v in mean.items():
            res = v / len(ranks)
            mean[k] = round(res, 2)
        mean = sorted(mean.items(), key=itemgetter(1), reverse=True)
        return mean


