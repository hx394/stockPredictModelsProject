import pandas as pd

# Load the uploaded files
aapl_file_path = 'path/to/AAPL=2024-11-14.csv'
pca_file_path = 'path/to/AAPL/pca_results.csv'

# Read the CSV files into DataFrames
aapl_data = pd.read_csv(aapl_file_path)
pca_data = pd.read_csv(pca_file_path)

# Convert PCA dates to datetime format and sort by date
pca_data['Date'] = pd.to_datetime(pca_data['Unnamed: 0'], format='%Y/%m/%d')
pca_data = pca_data.sort_values(by='Date').reset_index(drop=True)

# Convert AAPL dates to datetime format
aapl_data['timestamp'] = pd.to_datetime(aapl_data['timestamp'])

# Initialize a column for Principal Component 1 and 2
aapl_data['Principal Component 1'] = None
aapl_data['Principal Component 2'] = None

# Iterate over the PCA date ranges and assign values to AAPL data
for i in range(len(pca_data) - 1):
    # Get the current and next PCA date range
    current_date = pca_data.loc[i, 'Date']
    next_date = pca_data.loc[i + 1, 'Date']
    
    # Mask for AAPL dates within the range
    mask = (aapl_data['timestamp'] >= current_date) & (aapl_data['timestamp'] < next_date)
    
    # Assign PCA values to AAPL data
    aapl_data.loc[mask, 'Principal Component 1'] = pca_data.loc[i, 'Principal Component 1']
    aapl_data.loc[mask, 'Principal Component 2'] = pca_data.loc[i, 'Principal Component 2']

# Handle the last PCA date range
last_date = pca_data.iloc[-1]['Date']
aapl_data.loc[aapl_data['timestamp'] >= last_date, 'Principal Component 1'] = pca_data.iloc[-1]['Principal Component 1']
aapl_data.loc[aapl_data['timestamp'] >= last_date, 'Principal Component 2'] = pca_data.iloc[-1]['Principal Component 2']

# Display the updated AAPL data with the new columns
#import ace_tools as tools; tools.display_dataframe_to_user(name="Updated AAPL Data with PCA Columns", dataframe=aapl_data)

output_file_path = 'path/to/output'
aapl_data.to_csv(output_file_path, index=True)

print(f"data saved to file: {output_file_path}")