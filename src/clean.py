import pandas as pd

df = pd.read_csv("dataset/heart_disease_uci.csv")

df = df.drop(columns=["id", "dataset"])

df.loc[df["trestbps"] == 0, "trestbps"] = pd.NA

df["target"] = (df["num"] > 0).astype(int)

df = df.drop(columns=["num"])

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget:")
print(df["target"].value_counts())
df.to_csv("dataset/cleaned.csv", index=False)