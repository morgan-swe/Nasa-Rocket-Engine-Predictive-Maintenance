import pandas as pd

file_path = 'data/raw/test_FD001.txt'
data = pd.read_csv(file_path)
print(data.head())