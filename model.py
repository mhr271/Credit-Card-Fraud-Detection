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
from sklearn.metrics import PrecisionRecallDisplay

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

rf = Pipeline(
    steps=[
        ('model',RandomForestClassifier(n_estimators=30,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1
         ))
    ]
)
rf.fit(x_train,y_train)

rf_y_train_pred = rf.predict(x_train)
rf_y_test_pred = rf.predict(x_test)

rf_y_train_proba = rf.predict_proba(x_train)[:, 1]
rf_y_test_proba = rf.predict_proba(x_test)[:, 1]

rf_train_f1 = f1_score(y_train, rf_y_train_pred)
rf_train_pr_auc = average_precision_score(y_train, rf_y_train_proba)
rf_train_confusionmatrix = confusion_matrix(y_train, rf_y_train_pred)

rf_test_f1 = f1_score(y_test, rf_y_test_pred)
rf_test_pr_auc = average_precision_score(y_test, rf_y_test_proba)
rf_test_confusionmatrix = confusion_matrix(y_test, rf_y_test_pred)

rf_results = pd.DataFrame(
    [["Random Forest",
      rf_train_f1,
      rf_train_pr_auc,
      rf_train_confusionmatrix,
      rf_test_f1,
      rf_test_pr_auc,
      rf_test_confusionmatrix]],
    columns=[
        "Model",
        "Train f1",
        "Train PR-Auc",
        "Train ConfusionMatrix",
        "Test f1",
        "Test PR-Auc",
        "Test ConfusionMatrix"
    ]
)

results = pd.concat(
    [lr_results.drop(columns=["Train ConfusionMatrix", "Test ConfusionMatrix"]),
     rf_results.drop(columns=["Train ConfusionMatrix", "Test ConfusionMatrix"])],
    ignore_index=True
)
print(results)

print("\nLogistic Regression - Test Confusion Matrix:\n", lr_test_confusionmatrix)
print("\nRandom Forest - Test Confusion Matrix:\n", rf_test_confusionmatrix)

fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_test, lr_y_test_proba, name="Logistic Regression", ax=ax
)
PrecisionRecallDisplay.from_predictions(
    y_test, rf_y_test_proba, name="Random Forest", ax=ax
)

ax.set_title("Precision-Recall Curve: Logistic Regression vs Random Forest")
plt.savefig("pr_curve.png", dpi=200, bbox_inches="tight")
plt.show()

