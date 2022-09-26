import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class PlottingModel:
        def make_plot(self, model, score):
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            h = .02  # шаг регулярной сетки
            x0_min, x0_max = self.X[:, 0].min() - .5, self.X[:, 0].max() + .5
            x1_min, x1_max = self.X[:, 1].min() - .5, self.X[:, 1].max() + .5
            cm = plt.cm.RdBu
            xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
            if hasattr(model, "decision_function"):
                print(np.c_[xx0.ravel(), xx1.ravel()].shape)
                Z = model.decision_function(np.c_[xx0.ravel(), xx1.ravel()])
            elif hasattr(model, "predict_proba"):
                print(np.c_[xx0.ravel(), xx1.ravel()].shape)
                Z = model.predict_proba(np.c_[xx0.ravel(), xx1.ravel()])[:, 1]
            elif hasattr(model, "predict"):
                print(np.c_[xx0.ravel(), xx1.ravel()].shape)
                Z = model.predict(np.c_[xx0.ravel(), xx1.ravel()])
            Z = Z.reshape(xx0.shape)
            current_subplot = plt.subplot(1, 1, 1)
            current_subplot.contourf(xx0, xx1, Z, cmap=cm, alpha=.8)
            current_subplot.set_xlim(xx0.min(), xx0.max())
            current_subplot.set_ylim(xx0.min(), xx1.max())
            current_subplot.set_xticks(())
            current_subplot.set_yticks(())
            current_subplot.text(xx0.max() - .3, xx1.min() + .3, ('%.2f' % score),
                                 size=15, horizontalalignment='right')
            current_subplot.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright)
            current_subplot.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright, alpha=0.6)
            plt.show()
            return plt