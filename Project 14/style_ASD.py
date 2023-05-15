# Styling data--------------------------------------
data = 'Project 14\ASD_new.csv'
df_data = pd.read_csv(data)
# df_data.style.hide(axis="index")
# Create a Styler object from the DataFrame
styled_df = df_data.style

# Apply styles to the DataFrame
styled_df = styled_df.set_properties(**{'text-align': 'center'})

# Apply conditional formatting
styled_df = styled_df.background_gradient(cmap='Blues')

# Render the styled DataFrame
styled_df
# print(df_data)
