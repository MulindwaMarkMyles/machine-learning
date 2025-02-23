import scipy.io as sio
import pandas as pd

data = sio.loadmat("./Control_part1/Control_part1/v41p.mat")
v41p = data["v41p"]

df = pd.DataFrame(v41p)

with open("control_data_part_one_output.txt", "w") as file:
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

