import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


file_path = 'C:/Users/rain_/Desktop/stocks/vz-ratios-quarterly.xlsx'
data = pd.read_excel(file_path)


data.set_index("Date", inplace=True)
df = data.T  


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)


pca_df = pd.DataFrame(pca_data, columns=['Principal Component 1', 'Principal Component 2'], index=df.index)


pca_df.to_csv("pca_results.csv", index=True)


print("Explained variance ratio:", pca.explained_variance_ratio_)


