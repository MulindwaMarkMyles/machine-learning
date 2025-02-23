import scipy.io as sio
import pandas as pd

data = sio.loadmat("./ADHD_part1/ADHD_part1/v1p.mat")
v1p = data["v1p"]

df = pd.DataFrame(v1p)

with open("data_inspection_output.txt", "w") as file:
    file.write("Column Headers:\n")
    columns_list = [str(i) for i in df.columns.tolist()]
    file.write(", ".join(columns_list) + "\n\n")

    file.write("Data Preview:\n")
    file.write(df.head().to_string() + "\n\n")

    file.write("Data Information:\n")
    df.info(buf=file)
    file.write("\n\n")

    file.write("Data Description:\n")
    file.write(df.describe().to_string() + "\n\n")

    file.write("Missing Values:\n")
    file.write(df.isnull().sum().to_string() + "\n\n")

    file.write("Unique Values:\n")
    file.write(df.nunique().to_string() + "\n\n")

