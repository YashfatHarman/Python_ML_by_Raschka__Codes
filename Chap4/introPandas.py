import pandas as pd
from io import StringIO
import numpy as np

#csv_data = '''A, B, C, D
#1.0, 2.0, 3.0, 4.0
#5.0, 6.0, , 8.0
#0.0, 11.0, 12.0, '''


#df = pd.read_csv(StringIO(csv_data))
df = pd.read_csv("data.csv")

print(df)

print(df.isnull().sum())

print(df.values)


from sklearn.preprocessing import Imputer

imr = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

imr = imr.fit(df)

imputed_data = imr.transform(df.values)

print(imputed_data)
