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

from scipy.cluster.hierarchy import linkage

hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")
hc_single = linkage(df, "single")

print(hc_complete, hc_average, hc_single)

from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('hierarchical Classifaction')
plt.xlabel('Indexes')
plt.ylabel('Distance')

den = dendrogram(
    hc_complete,
    leaf_font_size=10
)
plt .show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage = "ward")
classes = cluster.fit_predict(df)

with_class = pd.DataFrame({"Eyaletler": df.index, "Classes" : classes})
df["class"] = classes
print(with_class)
print(df)
