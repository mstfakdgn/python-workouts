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

# print(df.head())
# print(df.isnull().sum())
# print(df.info())
# print(df.describe().T)
# df.hist(figsize = (10,10))
# plt.show()

kmeans = KMeans(n_clusters = 4).fit(df)
print(kmeans.n_clusters)
print(kmeans.cluster_centers_)

print(kmeans.labels_)

# #Visualizing
# kmeans2 = KMeans(n_clusters=2).fit(df)
# classes2 = kmeans2.labels_

# plt.scatter(df.iloc[:,0], df.iloc[:,1], c = classes2, s = 50, cmap="viridis")
# centres2 = kmeans2.cluster_centers_
# plt.scatter(centres2[:,0], centres2[:,1], c = 'black', s = 200, alpha=0.5)
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# # pip install --upgrade matplotlib
# # import mlp_toolkits
# kmeans3 = KMeans(n_clusters=3).fit(df)
# classes3 = kmeans3.labels_
# centres3 =kmeans3.cluster_centers_

# plt.rcParams['figure.figsize'] = (16,9)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2])
# plt.show()


# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], c = classes3)
# ax.scatter(centres3[:,0], centres3[:,1], centres3[:,2], marker='*', c='#050505', s = 1000)
# plt.show()


# class_data_frame = pd.DataFrame({"States": df.index, "Classes": classes3})
# print(class_data_frame[0:10])
# df["class"] = classes3
# print(df)



#Finding optimum class number
#Elbow
#pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(df)
visualizer.poof()