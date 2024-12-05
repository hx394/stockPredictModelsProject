import pandas as pd

# Load the updated PCA results file and Microsoft data file
msft_file_path = '/mnt/data/MSFT=2024-11-14.csv'
pca_file_path = '/mnt/data/pca_results.csv'

# Read the files into DataFrames
msft_data = pd.read_csv(msft_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert Microsoft dates to datetime format
msft_data['timestamp'] = pd.to_datetime(msft_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
msft_data['Principal Component 1'] = None
msft_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to Microsoft data
for i in range(len(pca_data) - 1):
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    mask = (msft_data['timestamp'] >= current_date) & (msft_data['timestamp'] < next_date)
    msft_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    msft_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
msft_data.loc[msft_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
msft_data.loc[msft_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Save the updated Microsoft data to a new file
msft_output_file_path = '/mnt/data/Updated_MSFT_Data_with_PCA.csv'
msft_data.to_csv(msft_output_file_path, index=False)

print(f"Updated Microsoft data saved to {msft_output_file_path}")