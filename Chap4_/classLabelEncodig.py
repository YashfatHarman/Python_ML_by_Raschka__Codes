import numpy as np
import pandas as pd

df = pd.DataFrame([ ["green", "M", 10.1, "class1"], ["red", "L", 13.5, "class2"], ["blue", "XL", 15.3, "class1"] ])

df.columns = ["color", "size", "price", "classlabel"]

print(df)

class_mapping = {label:idx for idx,label in enumerate(np.unique(df["classlabel"]))}
print(class_mapping)

df["classlabel"] = df["classlabel"].map(class_mapping)
print(df)

inv_class_mapping = {v:k for k,v in class_mapping.items()}
print(inv_class_mapping)

df["classlabel"] = df["classlabel"].map(inv_class_mapping)
print(df)

'''
Alternative: scikit-learn already has a LabelEncoder class that can take care of it directly.
'''

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

y = class_le.fit_transform(df["classlabel"].values)
print(y) 
