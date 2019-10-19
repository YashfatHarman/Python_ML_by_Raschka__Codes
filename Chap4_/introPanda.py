import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,4.0
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
#df = pd.read_csv(csv_data) #wont work as csv_data is not a real file, rather a string only

print(df)

#find the number of missing elements in each row
print(df.isnull().sum())

#drop rows with missing values
print(df.dropna())


#drop columns with missing values
print(df.dropna(axis=1))

#only drop rows where all columns are NaN
print(df.dropna(how="all"))

#drop rows that does not have at least 4 non-NaN values
print(df.dropna(thresh=4))

#only drop rows where NaN appear in specific columns (here, "C")
print(df.dropna(subset=["C"]))
print(df.dropna(subset=["C","D"]))

