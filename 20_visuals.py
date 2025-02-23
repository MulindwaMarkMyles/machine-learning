import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

df_mean = df.mean().reset_index()
df_mean.columns = ["Feature", "Mean Value"]

fig = px.bar_polar(df_mean, r="Mean Value", theta="Feature", color="Feature",
                   color_discrete_sequence=px.colors.sequential.Plasma)
fig.show()
