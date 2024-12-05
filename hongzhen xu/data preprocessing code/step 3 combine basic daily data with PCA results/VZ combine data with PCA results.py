import pandas as pd

# Load the updated PCA results file and Verizon data file
vz_file_path = '/mnt/data/VZ=2024-11-14.csv'
pca_file_path = '/mnt/data/pca_results.csv'

# Read the files into DataFrames
vz_data = pd.read_csv(vz_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert Verizon dates to datetime format
vz_data['timestamp'] = pd.to_datetime(vz_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
vz_data['Principal Component 1'] = None
vz_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to Verizon data
for i in range(len(pca_data) - 1):
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    mask = (vz_data['timestamp'] >= current_date) & (vz_data['timestamp'] < next_date)
    vz_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    vz_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
vz_data.loc[vz_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
vz_data.loc[vz_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Save the updated Verizon data to a new file
vz_output_file_path = '/mnt/data/Updated_VZ_Data_with_PCA.csv'
vz_data.to_csv(vz_output_file_path, index=False)

print(f"Updated Verizon data saved to {vz_output_file_path}")
