import pandas as pd

# Load the updated PCA results file and Tesla data file
tsla_file_path = '/mnt/data/TSLA=2024-11-14.csv'
pca_file_path = '/mnt/data/pca_results.csv'

# Read the files into DataFrames
tsla_data = pd.read_csv(tsla_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert Tesla dates to datetime format
tsla_data['timestamp'] = pd.to_datetime(tsla_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
tsla_data['Principal Component 1'] = None
tsla_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to Tesla data
for i in range(len(pca_data) - 1):
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    mask = (tsla_data['timestamp'] >= current_date) & (tsla_data['timestamp'] < next_date)
    tsla_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    tsla_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
tsla_data.loc[tsla_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
tsla_data.loc[tsla_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Save the updated Tesla data to a new file
tsla_output_file_path = '/mnt/data/Updated_TSLA_Data_with_PCA.csv'
tsla_data.to_csv(tsla_output_file_path, index=False)

print(f"Updated Tesla data saved to {tsla_output_file_path}")
