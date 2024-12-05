import pandas as pd

# Load the new PCA results file and Meta data file
meta_file_path = '/mnt/data/META=2024-11-14.csv'
pca_file_path = '/mnt/data/pca_results.csv'

# Read the files into DataFrames
meta_data = pd.read_csv(meta_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert Meta dates to datetime format
meta_data['timestamp'] = pd.to_datetime(meta_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
meta_data['Principal Component 1'] = None
meta_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to Meta data
for i in range(len(pca_data) - 1):
    # Get the current and next PCA date range
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    
    # Mask for Meta dates within the range
    mask = (meta_data['timestamp'] >= current_date) & (meta_data['timestamp'] < next_date)
    
    # Assign PCA values to Meta data
    meta_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    meta_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
meta_data.loc[meta_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
meta_data.loc[meta_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Save the updated Meta data to a new file
meta_output_file_path = '/mnt/data/Updated_META_Data_with_PCA.csv'
meta_data.to_csv(meta_output_file_path, index=False)

print(f"Updated Meta data saved to {meta_output_file_path}")
