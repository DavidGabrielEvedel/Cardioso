import pandas as pd

df = pd.read_csv("dataset/heart_disease_uci.csv")

print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nCategorical values:")

cols = ["sex", "dataset", "cp", "fbs", "restecg",
        "exang", "slope", "thal"]

for col in cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))
    
print("\nNumerical summary:")
print(df.describe())

print("\nUnique values:")

for col in ["fbs", "restecg", "exang", "ca", "num"]:
    print(f"\n{col}:")
    print(df[col].unique())
    
print("\nSuspicious trestbps values:")
print(df[df["trestbps"] <= 0][["age", "trestbps", "dataset", "num"]])

print("\nNegative oldpeak values:")
print(df[df["oldpeak"] < 0][["age", "oldpeak", "dataset", "num"]])

print("\nDuplicate rows:", df.duplicated().sum())

print("Duplicate IDs:", df["id"].duplicated().sum())