import pandas as pd

df = pd.read_csv("housing.data", header = None, sep = "\s+")

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

#sns.set(style = 'whitegrid', context = 'notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size = 1.5)
plt.show()

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ".2f", annot_kws = {"size": 15}, yticklabels = cols, xticklabels = cols)
plt.show()
