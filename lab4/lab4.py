from random import randrange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
def plot_clustering(df, clus_size=0):
    plt.scatter(df[labels==-1, 0], df[labels==-1, 1], s=10, c="black")
    plt.scatter(df[labels==0, 0], df[labels==0, 1], s=20, c="green")
    plt.scatter(df[labels==1, 0], df[labels==1, 1], s=20, c="yellow")
    plt.scatter(df[labels==2, 0], df[labels==2, 1], s=20, c="red")
    plt.scatter(df[labels==3, 0], df[labels==3, 1], s=20, c="pink")
    plt.scatter(df[labels==4, 0], df[labels==4, 1], s=20, c="blue")
    plt.scatter(df[labels==5, 0], df[labels==5, 1], s=20, c="purple")
    plt.scatter(df[labels==6, 0], df[labels==6, 1], s=20, c="orange")
    plt.scatter(df[labels==7, 0], df[labels==7, 1], s=20, c="brown")
    plt.scatter(df[labels==8, 0], df[labels==8, 1], s=20, c="olive")
    plt.xlabel('Salary in usd')
    plt.ylabel('Remote ratio')
    plt.title(f"Count of cluster:{clus_size}")
    plt.show()
data = pd.read_csv("../ds_salaries.csv")
df = data.loc[:, ['salary_in_usd', 'remote_ratio']].values
new_arr = np.array([df[i, 1]+randrange(30) for i in range(len(df))])
df[:, 1] = new_arr
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(df)
df = scaler.transform(df)
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(df)
plot_clustering(df, len(np.unique(labels)))