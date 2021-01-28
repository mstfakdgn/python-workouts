#Data cleaning, Data cleasing
#-Noisy Data
#-Missing Data
#-Outlier Data

#Data standardization, Feature scaling
# -0-1 transformation(Normalization)
# -z score transformation(Standardization)
# -logaritmic transform (Log transformation)

#Data reduction
# -Reducing observation count
# -reducing variable count 

#Variable transformation 
# -continues variables transformation 
# -categorical variable transformation












# #deleting missing data

# import seaborn as sns 
# import matplotlib.pyplot as plt

# df = sns.load_dataset("diamonds")
# df = df.select_dtypes(include = ['float64', 'int64'])
# # print(df)
# df = df.dropna()
# # print(df)


# #Finding outliers
# df_table = df["table"]
# sns.boxplot(x = df_table)
# print(df_table.shape)
# #we have determine a value to find outliers
# Q1 = df_table.quantile(0.25)
# Q3 = df_table.quantile(0.75)
# IQR = Q3 - Q1
# print(Q1, Q3, IQR)
# bottom_line = Q1 - 1.5*IQR
# upper_line = Q3 + 1.5*IQR
# print(bottom_line, upper_line)

# #riching outliers
# isOutlier =  ((df_table < bottom_line) | (df_table > upper_line))

# outliers = df_table[isOutlier]
# outlierIndexes = df_table[isOutlier].index

# print(outliers, outlierIndexes)

# # plt.show()




#solving outlier problem

# #Deleting
# import pandas as pd 
# # print(type(df_table))
# df_table = pd.DataFrame(df_table)
# print(df_table.shape)
# # ~ means that bring if condition is not true, like unless
# clean_df = df_table[~ ((df_table < bottom_line) | (df_table > upper_line)).any(axis = 1 )]
# print(clean_df.shape)



# #Filling with means
# table_mean = df["table"].mean()
# print(table_mean)

# # df_table[((df_table < bottom_line) | (df_table > upper_line))] = table_mean
# df_table[isOutlier] = table_mean

# print(df_table, df_table.shape)



# #baskılama pressure
# #outlier takes the value of upper line if it is bigger that upper line 
# #if itis lower than botton line it takes the value of bottom line

# df_table[df_table < bottom_line] = bottom_line
# df_table[df_table > upper_line] = upper_line
# print(df_table, df_table.shape)














# #muilti variable outlier observation
# #Local outlier factor
# #gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak buna göre aykırı değer olab,lecek değerleri tanıyabilmemize imkan sağlar 

# import seaborn as sns
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import LocalOutlierFactor

# diamonds = sns.load_dataset("diamonds")
# diamonds = diamonds.select_dtypes(include = ['float64', 'int64'])
# df = diamonds.copy()
# df = df.dropna()
# print(df)

# clf = LocalOutlierFactor(n_neighbors = 20, contamination=0.1)
# clf.fit_predict(df)
# df_scores = clf.negative_outlier_factor_

# #these are local oulier factor score
# # print(df_scores[0:10])
# print(np.sort(df_scores)[0:20])

# #assume that we accept tenth value as outlier level
# outlier_level = np.sort(df_scores)[10]
# isOutlier = df_scores > outlier_level


# # #Deleting
# # new_df = df[df_scores > outlier_level]
# # print(new_df)
# # outliers = df[df_scores < outlier_level]
# # print(outliers)


# # #Pressure
# # pressureValue = df[df_scores == outlier_level]
# # outliers = df[~isOutlier]
# # print(outliers)
# # rest = outliers.to_records(index = False)
# # print(rest)
# # rest[:] = pressureValue.to_records(index = False)
# # print(rest)

# # #convert to dataframe
# # df[~isOutlier] = pd.DataFrame(rest, index = df[~isOutlier].index)

# # print(df[~isOutlier])


























#missing data analize
# There are 3 types of missing data 
# fully accidental missing: There no connection with other attributes and has no structural problem 
# accidental missing: May happen conntected to other attributes
# non-accidental missing: We can not bypass it the reason may be structural


# to find out if it is accidental or not we have touse couple tests like 
# -visual teknik *****************
# -indipendent two sample T test
# -corralation test
#- Little's MCAR Test **************


#Missing data solutions

#deleting
# sample or variable deleting
#list base deleting(Listwise method)
#couplebase deletion(Pairwise method)


#assigning
# median, mean, middle
# most similar 
# external


# assigning according to guesses
# machine learning
# em
# multi assiging


# #solution
# import numpy as np
# import pandas as pd

# V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
# V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
# V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
# df = pd.DataFrame({
#     "V1":V1,
#     "V2":V2,
#     "V3":V3
# })
# print(df)
# #missing data count
# print(df.isnull().sum())
# #non missing data count
# print(df.notnull().sum())
# #all missing data
# print(df.isnull().sum().sum())

# # indexes that has at least one missing value
# print(df[df.isnull().any(axis = 1)])
# # indexes that has no missing data
# print(df[df.notnull().all(axis =1)])

# #same
# print(df[df["V1"].notnull() & df["V2"].notnull() & df["V3"].notnull()])


# #deleting directly missing valued indexes
# df.dropna(inplace=True)
# print(df)

#simple value assigning
# #fill with mean
# df["V1"].fillna(df["V1"].mean())
# df["V2"].fillna(0)

# #fill every attribute with its own mean *******************
# filledDf = df.apply(lambda x: x.fillna(x.mean()), axis=0)
# print(filledDf)



















# #missing value visualization !!pip install missingno
# import missingno as msno
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
# # V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
# # V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
# # df = pd.DataFrame({
# #     "V1":V1,
# #     "V2":V2,
# #     "V3":V3
# # })

# # #to show missing value chart
# # msno.bar(df);
# # #to show missing values with index
# # msno.matrix(df)
# # plt.show()

# planets = sns.load_dataset("planets")

# print(planets.isnull().sum())
# msno.bar(planets)
# msno.matrix(planets)
# msno.heatmap(planets)

# #from the grafics we can see that there is a connection between mass and orbiatl period 
# # whenever orbital period is nan there is no mass so mass is relying on orbital period
# plt.show()


























# #Deleting missing data
# import numpy as np
# import pandas as pd

# V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
# V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
# V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

# df = pd.DataFrame({
#     "V1":V1,
#     "V2":V2,
#     "V3":V3
# })
# print(df)

# # #Delete the row that has even one nan
# # df.dropna(inplace=True)

# # #Delete the row if every variable is nan
# # df.dropna(how ="all", inplace=True)

# # # Variable bas deleting meaning that if even a single nan in variable delete it 
# # df.dropna(axis = 1)

# # df['delete'] = np.nan
# # #Delete attributes if all values are nan
# # df.dropna(axis = 1, how="all", inplace=True)

# print(df)



















# #Simple variable assigning
# import numpy as np
# import pandas as pd

# V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
# V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
# V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

# df = pd.DataFrame({
#     "V1":V1,
#     "V2":V2,
#     "V3":V3
# })
# print(df)


# #Numeric values

# # df["V1"].fillna(df["V1"].mean(), inplace=True)
# # print(df)

# # #fill all nans with its attribute mean First way
# # filled_df = df.apply(lambda x: x.fillna(x.mean()), axis = 0)
# # print(filled_df)

# # #second way assing every nan to mean
# # df.fillna(df.mean()[:], inplace=True)
# # print(df)

# # # if we mix it
# # df.fillna(df.mean()["V1":"V2"], inplace=True)
# # print(df)
# # df["V3"].fillna(df["V3"].median(), inplace=True)
# # print(df)

# # #third way
# # filled_df = df.where(pd.notna(df), df.mean(), axis="columns")
# # print(filled_df)





















# #Categorical Variable Assigning (Kategorik değişken kırılımında değer atama)
# # her boşluğa attribute meanını atmak yerine diğer özellikleri aynı olan indexlerin meanlarını atmak daha başarılı olacaktır.
# import pandas as pd
# import numpy as np

# V1 = np.array([1,3,6,np.NaN, 7,1,np.NaN, 9,15])
# V2 = np.array([7, np.NaN, 5,8,12,np.NaN, np.NaN, 2,3])
# V3 = np.array([np.NaN, 12,5,6,14,7,np.NaN,2,31])
# V4 = np.array(["IT", "IT", "HR", "HR","HR","HR","HR","IT","IT"])

# df = pd.DataFrame({
#     "salary": V1,
#     "V2": V2,
#     "V3": V3,
#     "department": V4 
# })

# print(df)
# print(df.groupby("department")["salary"].mean())
# print(df.groupby("department")["V2"].mean())
# print(df.groupby("department")["V3"].mean())

# #fill empty salaries with departments own avarage salary
# print(df["salary"].fillna(df.groupby("department")["salary"].transform("mean")))

# #fill empty v2 with departments own avarage V2
# print(df["V2"].fillna(df.groupby("department")["V2"].transform("mean")))

# #fill empty v2 with departments own avarage V2
# print(df["V3"].fillna(df.groupby("department")["V3"].transform("mean")))

























# #Filling Empty categorical attributes
# import numpy as np
# import pandas as pd

# V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
# V4 = np.array(["IT",np.NaN, "HR", "HR", "HR", "HR", "HR", "IT", "IT"], dtype=object)

# df = pd.DataFrame({
#     "maas":V1,
#     "department":V4
# })

# print(df)

# # #fill the nan attributes with mod meaning that the most value that occures
# # print(df["department"].mode())
# # print(df["department"].mode()[0])
# # print(df["department"].fillna(df["department"].mode()[0]))

# # #fill the nan with next value
# # print(df["department"].fillna(method="bfill"))

# #fill the nan with before value
# print(df["department"].fillna(method="ffill"))





















# #Assigning to nan value according to guess using machine learning
# # pip install ycimpute

# import seaborn as sns
# import missingno as msno
# import numpy as np
# import pandas as pd

# df = sns.load_dataset("titanic")
# df = df.select_dtypes(include = ["float64", "int64"])
# print(df.head())
# print(df.isnull().sum())

# #kNN
# from ycimpute.imputer import knnimput

# var_names = list(df)
# n_df = np.array(df)

# print(var_names, n_df)

# dff = knnimput.KNN(k=4).complete(n_df)
# dff = pd.DataFrame(dff, columns = var_names)

# print(dff, dff.isnull().sum())


# #randomForest
# from ycimpute.imputer import iterforest

# var_names = list(df)
# n_df = np.array(df)

# print(var_names, n_df)

# dff = iterforest.IterImput().complete(n_df)
# dff = pd.DataFrame(dff, columns = var_names)

# print(dff, dff.isnull().sum())


# #EM
# from ycimpute.imputer import EM

# var_names = list(df)
# n_df = np.array(df)

# print(var_names, n_df)

# dff = EM().complete(n_df)
# dff = pd.DataFrame(dff, columns = var_names)

# print(dff, dff.isnull().sum())



























##standardization
# import numpy as np
# import pandas as pd

# V1 = np.array([1,3,6,5,7])
# V2 = np.array([7,7,5,8,12])
# V3 = np.array([6,12,5,6,14])

# df = pd.DataFrame({
#     "V1": V1,
#     "V2": V2,
#     "V3":V3
# })

# df = df.astype(float)
# print(df)

# from sklearn import preprocessing

# # print(preprocessing.scale(df))
# # print(df.mean())


# # #Normalization 0-1
# # print(preprocessing.normalize(df))

# # #min-max custom range
# # scaler = preprocessing.MinMaxScaler(feature_range = (10,40))
# # print(scaler.fit_transform(df))
























# #transformation

# import seaborn as sns
# import numpy as np

# df = sns.load_dataset("tips")
# print(df.head())

# from sklearn.preprocessing import LabelEncoder

# lbe = LabelEncoder()

# 0-1 transformation for example categorical attributes to 0 or 1 
# encoded_sex = lbe.fit_transform(df["sex"])
# df["encoded_sex"] = encoded_sex

# print(df)



# # 1 and others put choosen to 1 and 0 toall others
# df["new_day"] =np.where(df["day"].str.contains("Sun"),1,0)
# print(df)


# #multi choice chategorical change break impact on other attributes and output
# df["new_day"] = lbe.fit_transform(df["day"])
# print(df)
















# #One hot transformation and dummy variable trap ***************** the trap is they shouldn't be tell each other 
# import seaborn as sns 
# import numpy as np
# import pandas as pd

# df = sns.load_dataset("tips")
# print(df.head())

# # df_one_hot = pd.get_dummies(df, columns = ["sex"], prefix=["sex"])
# # print(df_one_hot)

# # df_one_hot_day = pd.get_dummies(df, columns = ["day"], prefix=["day"])
# # print(df_one_hot_day)





















# #data standardization and variable transformation
# import numpy as np
# import pandas as pd

# V1 = np.array([1,3,6,5,7])
# V2 = np.array([7,7,5,8,12])
# V3 = np.array([6,12,5,6,14])

# df = pd.DataFrame({
#     "V1":V1,
#     "V2":V2,
#     "V3":V3
# })

# df = df.astype(float)
# print(df)

# from sklearn import preprocessing

# print(preprocessing.scale(df))


# #Normalization
# print(preprocessing.normalize(df))


# #min-max transformation
# scaler = preprocessing.MinMaxScaler(feature_range=(10,20))
# print(scaler.fit_transform(df))


# #binarize transformation
# binarizer = preprocessing.Binarizer(threshold=5).fit(df)
# print(binarizer.transform(df))

















# # 0-1 transformation
# import seaborn as sns
# from sklearn import preprocessing

# tips = sns.load_dataset("tips")

# df = tips.copy()
# df_l = df.copy()

# print(df)

# # df_l["new_sex"] = df_l["sex"].cat.codes
# # print(df_l)

# lbe = preprocessing.LabelEncoder()
# df_l["new_sex"] = lbe.fit_transform(df_l["sex"])
# print(df_l)















# #0 and others
# import seaborn as sns
# from sklearn import preprocessing
# import numpy as np

# tips = sns.load_dataset("tips")

# df = tips.copy()
# df_l = df.copy()


# df_l["new_day"] = np.where(df_l["day"].str.contains("Sun"), 1, 0)

# print(df_l)















# #Multi Class Transform
# import seaborn as sns
# from sklearn import preprocessing
# import numpy as np

# tips = sns.load_dataset("tips")

# df = tips.copy()
# df_l = df.copy()

# lbe = preprocessing.LabelEncoder()
# df_l["new_day"] = lbe.fit_transform(df_l["day"])
# print(df_l)















# #One hot transformation
# import seaborn as sns
# from sklearn import preprocessing
# import numpy as np
# import pandas as pd

# tips = sns.load_dataset("tips")

# df = tips.copy()
# df_l = df.copy()

# one_wayed = pd.get_dummies(df_l, columns=["sex"], prefix=["sex"]).head()
# print(one_wayed)

# one_wayed_second = pd.get_dummies(one_wayed, columns=["day"], prefix=["day"]).head()
# print(one_wayed_second)





















# #contiuonus variables to categorical
# import seaborn as sns
# from sklearn import preprocessing
# import numpy as np
# import pandas as pd

# tips = sns.load_dataset("tips")
# df = tips.copy()
# df_l = df.copy()

# dff = df_l.select_dtypes(include = ["float64", "int64"])

# new = preprocessing.KBinsDiscretizer(n_bins = [3,2,2], encode = "ordinal", strategy="quantile").fit_transform(dff)
# print(new)
# print(df)













# #variables to index and indexes to variables
# import seaborn as sns
# from sklearn import preprocessing
# import numpy as np
# import pandas as pd

# tips = sns.load_dataset("tips")
# df = tips.copy()
# df_l = df.copy()

# df_l["new_variable"] = df_l.index
# df_l["new_variable"] = df_l["new_variable"] + 10
# print(df_l)