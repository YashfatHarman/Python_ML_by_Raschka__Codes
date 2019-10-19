import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data2.csv", header = None)

df.columns = ["color", "size", "price", "classLabel"]

print(df)

size_mapping = {
	'XL': 3,
	'L' : 2,
	'M' : 1,
}

df["size"] = df["size"].map(size_mapping)

print(df)	

X = df[["color","size","price"]].values
print(X)

color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

ohe = OneHotEncoder(categorical_features = [0])
arr = ohe.fit_transform(X).toarray()
print(arr)

withDummies = pd.get_dummies(df[["price","color","size"]])
print(withDummies)

