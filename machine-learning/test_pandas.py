import pandas as pd

df = pd.read_csv('test_data/water_usage_2025_02.csv')

Total = df['Usage Percentage'].sum()

print(Total)