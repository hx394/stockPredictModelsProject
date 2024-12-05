import pandas as pd

# List of uploaded file paths
file_paths = [
    '/mnt/data/Updated_AAPL_Data_with_PCA.csv',
    '/mnt/data/Updated_IBM_Data_with_PCA_2.csv',
    '/mnt/data/Updated_META_Data_with_PCA.csv',
    '/mnt/data/Updated_MSFT_Data_with_PCA.csv',
    '/mnt/data/Updated_NVDA_Data_with_PCA.csv',
    '/mnt/data/Updated_TSLA_Data_with_PCA.csv',
    '/mnt/data/Updated_VZ_Data_with_PCA.csv'
]

# Process each file by reversing the row order and saving
output_paths = []
for path in file_paths:
    data = pd.read_csv(path)
    reversed_data = data.iloc[::-1].reset_index(drop=True)
    output_path = path.replace('Updated', 'Reversed')
    reversed_data.to_csv(output_path, index=False)
    output_paths.append(output_path)

print("Reversed files saved to the following paths:")
for path in output_paths:
    print(path)
