import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")
df.columns = df.columns.astype(str)


sns.pairplot(df, diag_kind='hist', hue='0', vars=['0','1'])
plt.show()


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True,cmap='coolwarm')
plt.show()