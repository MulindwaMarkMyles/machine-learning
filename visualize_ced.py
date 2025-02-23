import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Standard-10-20-Cap19new/Standard-10-20-Cap19new.ced', delim_whitespace=True)

plt.figure(figsize=(10, 8))
plt.scatter(df['X'], df['Y'], c='blue', label='Electrode Positions')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Electrode Positions from .ced File')
plt.legend()
plt.grid(True)
plt.show()