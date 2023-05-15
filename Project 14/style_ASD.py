# Styling data--------------------------------------
import pandas as pd
data = 'Project 14\ASD_new.csv'

df_data = pd.read_csv(data)
pd.set_option('display.max_rows', None)  # Set option to display all rows
# print(df_data)


data_info = df_data.describe(include='all').T

# Add additional information to the codebook
data_info['data_type'] = df_data.dtypes
data_info['missing_values'] = df_data.isnull().sum()
data_info['unique_values'] = df_data.nunique()

# Print the codebook
print("Codebook:")
print(data_info)
