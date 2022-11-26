import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OneHotEncoder


def Sex_to_bool(sex):
    if sex == "male":
        return 0
    return 1


def part1():
    data = pd.read_csv('titanic.csv', index_col='PassengerId')
    data['Sex'] = data['Sex'].apply(Sex_to_bool)
    data = data.loc[(np.isnan(data['Pclass']) == False) & (np.isnan(data['Fare']) == False) & (np.isnan(data['Sex']) == False) &
                    (np.isnan(data['Survived']) == False)]
    corr = data[['Pclass', 'Fare', 'Age']]

    # определение целевой переменной
    y = data['Survived']
    # создание и обучение дерева решений
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(corr, y)
    # получение и распечатка важностей признаков
    importances = clf.feature_importances_

    print("Survived-Pclass ", importances[0])
    print("Survived-Fare ", importances[1])
    print("Survived-Sex ", importances[2])



part1()





