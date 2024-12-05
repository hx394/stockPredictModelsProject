import pandas as pd

# Reload the new IBM data file and PCA results
ibm_file_path = '/mnt/data/IBM=2024-11-14.csv'
pca_file_path = '/mnt/data/pca_results.csv'

# Read the files into DataFrames
ibm_data = pd.read_csv(ibm_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert IBM dates to datetime format
ibm_data['timestamp'] = pd.to_datetime(ibm_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
ibm_data['Principal Component 1'] = None
ibm_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to IBM data
for i in range(len(pca_data) - 1):
    # Get the current and next PCA date range
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    
    # Mask for IBM dates within the range
    mask = (ibm_data['timestamp'] >= current_date) & (ibm_data['timestamp'] < next_date)
    
    # Assign PCA values to IBM data
    ibm_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    ibm_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
ibm_data.loc[ibm_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
ibm_data.loc[ibm_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Save the updated IBM data to a new file
ibm_output_file_path = '/mnt/data/Updated_IBM_Data_with_PCA_2.csv'
ibm_data.to_csv(ibm_output_file_path, index=False)

print(f"Updated IBM data saved to {ibm_output_file_path}")
