import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/cleaned.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X train:", X_train.shape)
print("X test:", X_test.shape)
print("y train:", y_train.shape)
print("y test:", y_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

print("Numerical:", list(num_cols))
print("Categorical:", list(cat_cols))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

rf = Pipeline([
    ("pre", pre),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

xgb = Pipeline([
    ("pre", pre),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    ))
])

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

print("Random Forest trained")
print("XGBoost trained")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:, 1]
xgb_prob = xgb.predict_proba(X_test)[:, 1]


def show_results(name, pred, prob):

    print("\n" + "=" * 30)
    print(name)
    print("=" * 30)

    print("Accuracy :", round(accuracy_score(y_test, pred), 3))
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall   :", round(recall_score(y_test, pred), 3))
    print("F1 Score :", round(f1_score(y_test, pred), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_test, prob), 3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))


show_results("Random Forest", rf_pred, rf_prob)
show_results("XGBoost", xgb_pred, xgb_prob)

import pandas as pd

names = rf.named_steps["pre"].get_feature_names_out()

rf_imp = rf.named_steps["model"].feature_importances_

xgb_imp = xgb.named_steps["model"].feature_importances_

rf_f = pd.DataFrame({
    "feature": names,
    "importance": rf_imp
}).sort_values("importance", ascending=False)

xgb_f = pd.DataFrame({
    "feature": names,
    "importance": xgb_imp
}).sort_values("importance", ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_f.head(15))

print("\nXGBoost Feature Importance:")
print(xgb_f.head(15))

import joblib

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost.pkl")

print("\nModels saved!")