import numpy as np
import pandas as pd
import os

def dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

os.chdir("C:/Users/aceni/Documents/Uni Work/Dissertation/Benchmarks/CSV")

path = 'C1_2_1.csv'
df = pd.read_csv(path, index_col='CUST_NO')

n = len(df)

d = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        x1 = df.iloc[i,0]
        x2 = df.iloc[j,0]
        y1 = df.iloc[i,1]
        y2 = df.iloc[j,1]
        dij = dist(x1,y1,x2,y2)
        d[i,j] = np.inf if i == j else dij

os.chdir("C:/Users/aceni/Documents/Uni Work/Dissertation/Benchmarks/matrix")

save = 'C1_2_1.npy'

with open(save, 'wb') as f:
    np.save(f, d)

