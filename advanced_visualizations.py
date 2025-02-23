import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from pandas.plotting import parallel_coordinates

df = pd.read_csv("cleaned_dataset.csv")
df.columns = df.columns.astype(str)


parallel_coordinates(df, '1')
plt.show()
