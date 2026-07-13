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

lr_y_train_proba = lr.predict_proba(x_train)[:, 1]
lr_y_test_proba = lr.predict_proba(x_test)[:, 1]

lr_train_f1 = f1_score(y_train, lr_y_train_pred)
lr_train_pr_auc = average_precision_score(y_train, lr_y_train_proba)
lr_train_confusionmatrix = confusion_matrix(y_train, lr_y_train_pred)

lr_test_f1 = f1_score(y_test, lr_y_test_pred)
lr_test_pr_auc = average_precision_score(y_test, lr_y_test_proba)
lr_test_confusionmatrix = confusion_matrix(y_test, lr_y_test_pred)

lr_results = pd.DataFrame(
    [["Logistic Regression", lr_train_f1, lr_train_pr_auc, lr_train_confusionmatrix,
      lr_test_f1, lr_test_pr_auc, lr_test_confusionmatrix]],
    columns=["Model", "Train f1", "Train PR-Auc", "Train ConfusionMatrix",
             "Test f1", "Test PR-Auc", "Test ConfusionMatrix"],
)
print(lr_results)

rf = Pipeline(
    steps=[
        ('model',RandomForestClassifier(n_estimators=30,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1
         ))
    ]
)
rf.fit(x_train,y_train)
