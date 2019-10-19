import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

df_wine = pd.read_csv("wine.data", header = None)

print(df_wine.shape)


df_wine.columns = ["Class label", "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

print("Class labels",np.unique(df_wine["Class label"]))

print(df_wine.head())


X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


#now do feature scaling

#minmax scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print(X_train_norm[:5,:])


#standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train_std[:5,:])

