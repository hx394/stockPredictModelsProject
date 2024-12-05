import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


file_path = 'C:/Users/rain_/Desktop/code/stocks/meta-ratios-quarterly.xlsx'
data = pd.read_excel(file_path)


data_values = data.drop(columns=["Date"]).T


pca = PCA()
pca.fit(data_values)


explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Individual explained variance')
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='s', label='Cumulative explained variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot (Elbow Method) for PCA')
plt.legend()
plt.grid(True)
plt.show()
