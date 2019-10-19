import pandas as pd
import numpy as np

df = pd.read_csv("data2.csv", header = None)

df.columns = ["color", "size", "price", "classlabel"]

print(df)
#print(df.values)

size_mapping = {
	'XL': 3,
	'L' : 2,
	'M' : 1,
}

df["size"] = df["size"].map(size_mapping)
print(df)

inv_size_mapping = {v:k for k,v in size_mapping.items()}
#df["size"] = df["size"].map(inv_size_mapping)
#print(df)

class_mapping = {label:idx for idx,label in enumerate(np.unique(df["classlabel"])) }

df["classlabel"] = df["classlabel"].map(class_mapping)
print(df)

