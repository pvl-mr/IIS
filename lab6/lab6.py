import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
data = pd.read_csv("../ds_salaries.csv")
y = data['company_size']
data = data.drop(['company_size'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.1, random_state=42)
cols_to_lowercase = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'employee_residence', 'company_location']
for col in cols_to_lowercase:
    data[col] = data[col].str.lower()
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), cols_to_lowercase),
    remainder='passthrough'
)
mlp = MLPClassifier(random_state=321,
                    solver="sgd",
                    activation="tanh",
                    alpha=0.01,
                    hidden_layer_sizes=(2, ),
                    max_iter=2000,
                    tol=0.00000001)
pipe = make_pipeline(column_trans, mlp)
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
score = accuracy_score(y_test, predictions)
print(score)