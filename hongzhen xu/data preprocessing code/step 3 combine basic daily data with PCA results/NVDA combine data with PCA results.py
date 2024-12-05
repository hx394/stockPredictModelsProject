import pandas as pd

# Load the updated PCA results file and NVIDIA data file
nvda_file_path = '/mnt/data/NVDA=2024-11-14.csv'
pca_file_path = '/mnt/data/pca_results.csv'

# Read the files into DataFrames
nvda_data = pd.read_csv(nvda_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert NVIDIA dates to datetime format
nvda_data['timestamp'] = pd.to_datetime(nvda_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
nvda_data['Principal Component 1'] = None
nvda_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to NVIDIA data
for i in range(len(pca_data) - 1):
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    mask = (nvda_data['timestamp'] >= current_date) & (nvda_data['timestamp'] < next_date)
    nvda_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    nvda_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
nvda_data.loc[nvda_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
nvda_data.loc[nvda_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Save the updated NVIDIA data to a new file
nvda_output_file_path = '/mnt/data/Updated_NVDA_Data_with_PCA.csv'
nvda_data.to_csv(nvda_output_file_path, index=False)

print(f"Updated NVIDIA data saved to {nvda_output_file_path}")
