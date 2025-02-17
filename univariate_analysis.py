import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_dataset.csv")

df.hist(figsize=(20,20))
plt.show()

plt.figure(figsize=(20,20))
sns.boxplot(data=df)
plt.show()