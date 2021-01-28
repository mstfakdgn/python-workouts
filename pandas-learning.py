import pandas as pd
import numpy as np
import seaborn as sns

# a = pd.Series([1,2,3,4,5])
# print(a)
# print(type(a))
# print(a.axes)
# print(a.dtype)
# print(a.size)
# print(a.ndim)
# print(a.values)
# print(a.head(2))
# print(a.tail(2))

#index naming
# b = pd.Series([99,222,332,94,55, 986], index =[0,2,4,6,8,10])
# print(b)
# c = pd.Series([99,222,332,94,55, 986], index =["a","b","c","d","e","f"])
# print(c, c["a"])
# print(c["a":"d"])
# #creating list from dictionary
# dic = pd.Series({"reg":10, "log":11, "cart":12})
# print(dic)

# #concat series
# concated = pd.concat([b,c])
# print(concated)














#element operations
# import numpy as np
# import pandas as pd

# a = np.array([11,22333,444,55555,66666])
# seri = pd.Series(a)
# print(a)
# print(seri[1])
# print(seri[0:3])

# new_seri = pd.Series([121,200,150,99], index = ["reg", "loj", "cart", "rf"])
# print(new_seri)
# print(new_seri.index)
# print(new_seri.keys)
# print(list(new_seri.items()))
# print(new_seri.values)

#element retrieve
# print("reg" in new_seri)
# print(new_seri[["reg", "loj"]])
# new_seri["reg"] = "new"
# print(new_seri)
# print(new_seri["reg":"cart"])

















#Dataframe
# import pandas as pd
# import numpy as np

# l = (1,2,39,67,90)

# dataFrame = pd.DataFrame(l, columns=["attribute_name"])
# print(dataFrame)

# m = np.arange(1,10).reshape((3,3))
# dataFrame2 = pd.DataFrame(m, columns=["att1","att2","att3"])
# print(dataFrame2)
# print(dataFrame2.columns)
# dataFrame2.columns = ("new_att1", "new_att2", "new_att3")
# print(dataFrame2)
# print(type(dataFrame2))
# print(dataFrame2.axes)
# print(dataFrame2.shape)
# print(dataFrame2.ndim)
# print(dataFrame2.size)
# print(type(dataFrame2.values))
# print(dataFrame2.head(2))
# print(dataFrame2.tail(2))

# a = np.array([1,2,3,4,5,6])
# pd.DataFrame(a, columns = ["deg1"])
# print(a)
















#DataFrame element operations
# import numpy as np
# import pandas as pd

# s1 = np.random.randint(10, size = 5)
# s2 = np.random.randint(10, size = 5)
# s3 = np.random.randint(10, size = 5)

# dictionary = {"att1": s1, "att2": s2, "att3":s3}
# # print(dictionary)
# df = pd.DataFrame(dictionary)
# # print(df)
# # print(df[0:1])
# # print(df.index)
# df.index = ["a","b","c","d","e"]
# # print(df)
# # print(df["c":"e"])

# #deleting without efecting orginal
# ddf = df.drop("b", axis = 0)
# print(ddf)
# #deleting inplace
# df.drop("c", axis = 0, inplace = True)
# print(df)
# #fancy deleting
# l = ["a","e"]
# dddf = df.drop(l)
# print(dddf)

# print("att1" in df)

# # l = ["att1","att2","att3","att4"]
# # for i in l:
# #     print(i in df)

# df["att4"] = df["att1"]/df["att2"]
# print(df)
# df.drop("att4", axis = 1, inplace = True)
# print(df)

# l = ["att1","att2"]
# df.drop(l, axis = 1, inplace = True)
# print(df)



















#observation and attribute choosing loc, iloc
# import numpy as np
# import pandas as pd

# m = np.random.randint(1,30, size=(10,3))
# df = pd.DataFrame(m, columns = ["att1", "att2", "att3"])
# print(df)

#loc we make choose from its declared time
# print(df.loc[0:3])
#iloc normal choose
# print(df.iloc[0:3])
# print(df.iloc[0,0])
# print(df.iloc[:3,:2])

# print(df.loc[0:3, "att3"])
# print(df.iloc[0:3]["att3"])

















#Conditional element operations
# import numpy as np
# import pandas as pd

# m = np.random.randint(1,30, size=(10,3))
# df = pd.DataFrame(m, columns = ["att1", "att2", "att3"])
# print(df)
# print(df.att1)
# print(df[df.att1 > 20])
# print(df[df.att1 < 20])
# print(df[df.att1 < 20]["att1"])
# print(df[df.att1 > 20]["att1"])

# print(df[(df.att1 >15) & (df.att3 > 15)])
# print(df.loc[(df.att1 >15) & (df.att3 > 15), ["att1","att3"]])


















#join operations
# m = np.random.randint(1,30, size= (5,3))
# df1 = pd.DataFrame(m, columns=["att1","att2","att3"]) 
# print(df1)
# df2 = df1+ 100
# print(df2)

# concated = pd.concat([df1,df2])
# print(concated)

# concated = pd.concat([df1,df2], ignore_index=True)
# print(concated)

# if column attributes are different concated with NANs
# df2.columns = ["att1", "att2", "deg3"]
# print(df2)
# concated2 = pd.concat([df1,df2])
# print(concated2)

#with inner join
# innerJoined = pd.concat([df1,df2], join="inner")
# print(innerJoined)

#not working on new pandas but meand use df1 attributes
# concated = pd.concat([df1,df2], join_axes = [df1.columns])
# print(concated)

#not working on new pandas but meand use df2 attributes
# concated = pd.concat([df1,df2], join_axes = [df2.columns], ignore_index=True)
# print(concated)

















#advance join operations
# df1 = pd.DataFrame({'employees': ['Ali', 'Veli', 'Ayşe', 'Fatma'], 'group': ['Account', 'Engineering', 'Engineering', 'HR']})
# df2 = pd.DataFrame({'employees': ['Ayşe', 'Ali', 'Veli', 'Fatma'], 'start': [2010, 2009, 2014, 2019]})
# print(df1)
# print(df2)
# merged = pd.merge(df1,df2)
# print(merged)
# merged2 = pd.merge(df1,df2, on = "employees")
# print(merged2)

#many to one
# df3 = pd.merge(df1,df2)
# df4 = pd.DataFrame({'group': ['Account', 'Engineering', 'HR'], 'manager' : ['Caner', 'Mustafa', 'Berkcan']})
# print(df4)
# print(df3)

# merged3 = pd.merge(df3,df4)
# print(merged3)

#mant to many
# df5 = pd.DataFrame({'group' : [ 'Account', 'Account', 'Engineering', 'Engineering', 'HR', 'HR'], 'talents': ['Mat', 'Excel', 'Codding','Linux', 'Excel', 'Management']})
# print(df5)
# print(df1)
# merged4 = pd.merge(df5,df1)
# print(merged4)





















#Aggregation and Grouping
# import seaborn as sns

# df = sns.load_dataset("planets")
# print(df.mean())
# print(df["mass"].mean())
# print(df["mass"].max())
# print(df["mass"].min())
# print(df["mass"].sum())
# print(df["mass"].std())
# print(df["mass"].var())
# print(df.describe().T)
# print(df.dropna().describe().T)
















#Grouping
# import seaborn as sns
# # df = pd.DataFrame({'groups' : ['A','B','C','A','B','C'], 'data': [10,11,52,23,43,55]}, columns=['groups', 'data'])
# # print(df)
# # grouped = df.groupby("groups")
# # print(grouped)
# # print(grouped.mean())
# # print(grouped.sum())

# df = sns.load_dataset("planets")
# print(df)
# print(df.groupby("method")["orbital_period"].mean())
# print(df.groupby("method")["mass"].mean())
# print(df.groupby("method")["mass"].describe())





















#aggregate
# df = pd.DataFrame({
#         'groups': ['A', 'B', 'C', 'A', 'B', 'C'], 
#         'att1' : [10,23,33,22,11,99], 
#         'att2': [100,253,333,262,111,969]}, 
#         columns = ['groups', 'att1', 'att2']
#     )
# print(df)
# #aggregate
# print(df.groupby("groups").mean())
# print(df.groupby("groups").aggregate(["min", np.median, max]))
# print(df.groupby("groups").aggregate({"att1": "min", "att2" : "max"}))
























#filtering custom filterin with custom func
# df = pd.DataFrame({
#         'groups': ['A', 'B', 'C', 'A', 'B', 'C'], 
#         'att1' : [10,23,33,22,11,99], 
#         'att2': [100,253,333,262,111,969]}, 
#         columns = ['groups', 'att1', 'att2']
#     )
# print(df)

# def filter_func(x):
#     return x["att1"].std() > 9

# filteredDf = df.groupby("groups").filter(filter_func)

# print(df.groupby("groups").std())
# print(filteredDf)






















#transform
# df = pd.DataFrame({
#         'groups': ['A', 'B', 'C', 'A', 'B', 'C'], 
#         'att1' : [10,23,33,22,11,99], 
#         'att2': [100,253,333,262,111,969]}, 
#         columns = ['groups', 'att1', 'att2']
#     )
# # print(df)
# # print(df["att1"]*9)
# df_a = df.iloc[:,1:3]
# print(df_a.transform(lambda x: (x-x.mean()/ x.std())))























#apply
# df = pd.DataFrame({
#         'att1' : [10,23,33,22,11,99], 
#         'att2': [100,253,333,262,111,969]}, 
#         columns = ['att1', 'att2']
#     )
# print(df)
# print(df.apply(np.sum))
# print(df.apply(np.mean))

#with groups
# df = pd.DataFrame({
#         'groups': ['A', 'B', 'C', 'A', 'B', 'C'], 
#         'att1' : [10,23,33,22,11,99], 
#         'att2': [100,253,333,262,111,969]}, 
#         columns = ['groups', 'att1', 'att2']
#     )
# print(df)
# print(df.groupby('groups').apply(np.sum))
# print(df.groupby('groups').apply(np.mean))

















#pivot tables
# titanic = sns.load_dataset("titanic")
# print(titanic.head())

# print(titanic.groupby("sex")["survived"].mean())
# print(titanic.groupby(["sex", "class"])[["survived"]].aggregate("mean").unstack())

# #pivot table with pivot
# print(titanic.pivot_table("survived", index = "sex", columns = "class"))

# age = pd.cut(titanic["age"], [0,18,90])
# print(age.tail())
# print(titanic.pivot_table("survived", index = ["sex", age], columns = "class"))













# # reading external data from local
# csv = pd.read_csv("reading_data/ornekcsv.csv", sep = ";")
# # print(csv)
# txt = pd.read_csv("reading_data/duz_metin.txt")
# # print(txt)
# xlsx = pd.read_excel("reading_data/ornekx.xlsx")
# # print(xlsx)

# #read external data from internet
# tips = pd.read_csv("reading_data/tips.html")
# print(tips)

















