# Styling data--------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
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
# print("Codebook:")
print(data_info)
# # Plot a histogram of a numerical column
# x = 'id'
# y = 'age'

# # Create a scatter plot
# plt.bar(x, y)

# # Set labels and title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Scatter Plot')

# # Show the plot
# plt.show()


#--------------------------------------------------------------------





# import pandas as pd
# from sklearn.datasets import load_breast_cancer

# # Load the breast cancer dataset
# data = load_breast_cancer()

# # Convert the dataset into a DataFrame
# df_data = pd.DataFrame(data.data, columns=data.feature_names)

# # Get basic information about the DataFrame
# data_info = df_data.describe(include='all').T

# # Add additional information to the codebook
# data_info['data_type'] = df_data.dtypes
# data_info['missing_values'] = df_data.isnull().sum()
# data_info['unique_values'] = df_data.nunique()

# # Print the codebook
# print("Codebook:")
# print(data_info)