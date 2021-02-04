from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans

df = pd.read_csv('../../reading_data/USArrests.csv').copy()

df.index=df.iloc[:,0]
df = df.iloc[:, 1:5]
df.index.name = None
print(df.head())

from sklearn.preprocessing import StandardScaler

df_scaled = StandardScaler().fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df_scaled)

new_df = pd.DataFrame(data=pca_fit, columns=["first", "second"])
print(new_df.head())

print('Variance:',pca.explained_variance_ratio_)

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()