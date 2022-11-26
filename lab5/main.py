import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
data = pd.read_csv("../ds_salaries.csv")
y = data['salary_in_usd']
data = data.drop('db_id', axis=1)
data = data.drop('salary_in_usd', axis=1)
data = data.drop('salary', axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.1, random_state=42)
cols_to_lowercase = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'employee_residence', 'company_location', 'company_size']
#data[cols_to_lowercase] = data[cols_to_lowercase].str.lower()
for col in cols_to_lowercase:
    data[col] = data[col].str.lower()
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), cols_to_lowercase),
    remainder='passthrough'
)
ridge_regression = Ridge(alpha=1, random_state=241)
pipe = make_pipeline(column_trans, ridge_regression)
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
