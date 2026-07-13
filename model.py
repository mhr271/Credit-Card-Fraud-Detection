import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

df = pd.read_csv("creditcard.csv")
print(df)

y = df["Class"]
x = df.drop("Class",axis=1)

x_train , x_test, y_train , y_test = train_test_split(x,y
    ,test_size=0.2, random_state=42 , stratify=y)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

lr = Pipeline(
    steps=[
    ('scale',StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced',max_iter=200))
    ]
)
lr.fit(x_train,y_train)

lr_y_train_pred = lr.predict(x_train)
lr_y_test_pred = lr.predict(x_test)
