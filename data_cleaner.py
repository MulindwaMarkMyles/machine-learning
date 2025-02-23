import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import numpy as np
from scipy import stats

data = sio.loadmat("./Control_part1/Control_part1/v41p.mat")
v1p = data["v41p"]


df = pd.DataFrame(v1p)

# df.dropna(inplace=True)
#
# df.fillna(df.mean(), inplace=True)

df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

scaler = StandardScaler()
df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))

df = pd.get_dummies(df, drop_first=True)

df.to_csv('cleaned_dataset_control.csv', index=False)
