# import seaborn as sns

# planets = sns.load_dataset("planets")
# # print(planets.head())

# #story of dataSet
# df = planets.copy()
# # print(df.tail())

# #dataset structural info
# # print(df.info())
# # print(df.dtypes)

# #converting to catyegorical
# import pandas as pd

# df.method = pd.Categorical(df.method)
# print(df.dtypes)
# print(df.head())

















# import seaborn as sns

# planets = sns.load_dataset("planets")
# df = planets.copy()
# print(df.head())
# print(df.shape)
# print(df.columns)

# #ignores empty values and categorical attributes
# print(df.describe().T)
# print(df.describe(include = "all").T)





















# import seaborn as sns

# planets = sns.load_dataset("planets")
# df = planets.copy()
# print(df.head())

#check non exist attribıutes
# print(df.isnull().values.any()) #true or false

#which attributes has how many null
# print(df.isnull().sum())

#put 0 to all nan exist values
# df['mass'].fillna(0, inplace = True)
# print(df.isnull().sum())

# df['orbital_period'].fillna(df.orbital_period.mean(), inplace = True)
# print(df.isnull().sum())

#mean to every null attribte in df
# df.fillna(df.mean(), inplace = True)
# print(df.isnull().sum())

















#categorical attributes exemining
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# planets = sns.load_dataset("planets")
# df = planets.copy()
# print(df.head())

# cat_df = df.select_dtypes(include = ["object"])
# print(cat_df)

# print(cat_df.method.unique())
# print(cat_df["method"].value_counts().count())
# print(cat_df["method"].value_counts())

# df["method"].value_counts().plot.barh()

# plt.show()


















#non-discreate attributes
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# planets = sns.load_dataset("planets")
# df = planets.copy()
# # print(df.head())

# df_num = df.select_dtypes(include = ["float64", "int64"])
# # print(df_num)
# # print(df_num.describe().T)
# # print(df_num["distance"].describe().T)
# print("Avarage:" + str(df_num["distance"].mean()))
# print("Number of Full Observations:" + str(df_num["distance"].count()))
# print("Maximum Value:" + str(df_num["distance"].max()))
# print("Minimum Value:" + str(df_num["distance"].min()))
# print("Median:" + str(df_num["distance"].median()))
# print("Standart Deviation:" + str(df_num["distance"].std()))

















# #Scatter Chart
# #Barplot uses for visualizing categorical attributes
# import seaborn as sns
# import matplotlib.pyplot as plt

# diamonds = sns.load_dataset("diamonds")
# df = diamonds.copy()
# # print(df.head())
# # print(df.info())
# # print(df.describe().T)
# # print(df["cut"].value_counts()) #categorical
# # print(df["color"].value_counts()) #categorical

# #diamond categorical attributes are ordinal not nominal
# #ordinal
# # ordinal order is not true we have to give the sort
# from pandas.api.types import CategoricalDtype
# print(df.cut.head())
# df.cut = df.cut.astype(CategoricalDtype(ordered = True))
# print(df.dtypes)
# print(df.cut.head(1))

# cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
# df.cut = df.cut.astype(CategoricalDtype(cut_categories, ordered=True))
# print(df.dtypes)
# print(df.cut.head(1))



# #grafic barplot
# # df["cut"].value_counts().plot.barh().set_title("Class Frequency of Cut Attribute")
# # plt.show()

# #with seaborn
# sns.barplot(x = "cut", y = df.cut.index, data = df)
# plt.show()
























# #Evaluating attributes together
# import seaborn as sns
# from pandas.api.types import CategoricalDtype
# import matplotlib.pyplot as plt

# diamonds = sns.load_dataset("diamonds")
# df = diamonds.copy()
# cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
# df.cut = df.cut.astype(CategoricalDtype(cut_categories, ordered=True))
# # print(df.cut.head)
# # print(df.head)

# # sns.catplot(x = "cut", y="price", data= df)

# #cross validated
# sns.barplot(x = "cut", y ="price", hue="color", data = df)

# print(df.groupby(["cut", "color"])["price"].mean())
# plt.show()

























# #Histogram and Density for numerc attributes
# import seaborn as sns
# import matplotlib.pyplot as plt

# diamonds = sns.load_dataset("diamonds")
# df = diamonds.copy()
# print(df.head())
# #if kde true ise yoğunluk grafiği ile birlikte çizilir
# #y değerleri 0 1 arası ölçeklenir
# # sns.distplot(df.price, bins = 10, kde = True)
# # sns.distplot(df.price, hist = False)
# # plt.show()

# # print(df["price"].describe())
# sns.kdeplot(df.price, shade = True)
# plt.show()


















# #Cross Validation for Histogram
# import seaborn as sns
# import matplotlib.pyplot as plt

# diamonds = sns.load_dataset("diamonds")
# df = diamonds.copy()
# # print(df.head())

# sns.kdeplot(df.price, shade=True)
# (sns
#     .FacetGrid(df,
#         hue="cut",
#         height=5,
#         xlim = (0,10000))
#     .map(sns.kdeplot, "price", shade=True)
#     .add_legend()
# )
# sns.catplot(x="cut", y="price", hue="color", kind="point", data = df)
# plt.show()

















# #boxplot
# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# df = tips.copy()
# # print(df.head())
# # print(df.describe().T)

# print(df["sex"].value_counts())
# print(df["smoker"].value_counts())
# print(df["time"].value_counts())
# print(df["day"].value_counts())


# # sns.barplot(x = "size", y ="tip", hue="sex", data = df)
# # sns.barplot(x = "size", y ="tip", hue="smoker", data = df)

# sns.boxplot(x=df["total_bill"])
# sns.boxplot(x=df["total_bill"], orient = "v")
# plt.show()


















##Boxplot Cross
# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# df = tips.copy()
# print(df.head())
# print(df.describe().T)
#print(df["day"].value_counts())

#Which days we earn more
# sns.kdeplot(df.price, shade=True)
# (sns
#     .FacetGrid(df,
#         hue="day",
#         height=5,
#         xlim = (0,100))
#     .map(sns.kdeplot, "total_bill", shade=True)
#     .add_legend()
# )
# sns.boxplot(x = "day", y = "total_bill", data = df)
# plt.show()

#Do we earn more in day or evening
# sns.boxplot(x="time", y="total_bill", data=df)
# plt.show()


#Number of person with earning
# sns.boxplot(x = "size", y="total_bill", data=df)
# plt.show()

#Sex with total bill according to days
# sns.boxplot(x = "day", y="total_bill",hue="sex", data=df)
# plt.show()





















# #Violin grafic
# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# df = tips.copy()

# sns.catplot(y="total_bill", kind="violin", data=df)
# plt.show()

# sns.catplot(x="day", y="total_bill", kind="violin", data=df)
# plt.show()

# sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", data=df)
# plt.show()






















# #Corroletion Grafics Scatterplot is relation between two numerical attributes
# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# df = tips.copy()
# print(df.head())

# sns.scatterplot(x ="total_bill", y ="tip", data=df)
# plt.show()

# sns.scatterplot(x ="total_bill", y ="tip", hue="time", data=df)
# plt.show()

# sns.scatterplot(x ="total_bill", y ="tip", hue="time", style="time", data=df)
# plt.show()

# sns.scatterplot(x ="total_bill", y ="tip", hue="day", style="day", data=df)
# plt.show()

# sns.scatterplot(x="total_bill", y="tip", hue="size", size="size", data = df);
# plt.show()

















# #Lineer relation
# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# df = tips.copy()

# sns.lmplot(x = "total_bill", y="tip", hue="smoker", data=df)
# plt.show()

# sns.lmplot(x = "total_bill", y="tip", hue="smoker",col="time", data=df)
# plt.show()

# sns.lmplot(x = "total_bill", y="tip", hue="smoker",col="time", row ="sex", data=df)
# plt.show()



















# #Scatterplot Matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# iris = sns.load_dataset("iris")
# df = iris.copy()
# print(df.head())
# print(df.dtypes)
# print(df.shape)

# sns.pairplot(df)
# plt.show()

# sns.pairplot(df, hue="species")
# plt.show()

# sns.pairplot(df, hue="species", markers = ["o","s","D"])
# plt.show()

# sns.pairplot(df, kind="reg")
# plt.show()

# sns.pairplot(df, hue="species", kind="reg")
# plt.show()




















# #Heat Meap
# import seaborn as sns
# import matplotlib.pyplot as plt

# flights = sns.load_dataset("flights")
# df = flights.copy()
# # print(df)
# # print(df.describe().T)
# # print(df.shape)
# # print(df["year"].describe())

# df = df.pivot("month", "year", "passengers")
# # print(df)

# # sns.heatmap(df)
# # plt.show()

# # sns.heatmap(df, annot = True, fmt="d")
# # plt.show()

# # sns.heatmap(df, annot = True, fmt="d", linewidths = .5)
# # plt.show()























# #Line Grafic
# import seaborn as sns
# import matplotlib.pyplot as plt

# fmri = sns.load_dataset("fmri")
# df = fmri.copy()
# print(df)
# print(df.shape)

# print(df["timepoint"].describe())
# print(df["signal"].describe())

# print(df.groupby("timepoint")["signal"].count())
# print(df.groupby("signal").count()) # unique

# print(df.groupby("timepoint")["signal"].describe())

# sns.lineplot(x = "timepoint", y="signal", data = df)
# plt.show()

# sns.lineplot(x = "timepoint", y="signal", hue ="event",data = df)
# plt.show()

# sns.lineplot(x = "timepoint", y="signal", hue ="event", style="event",data = df)
# plt.show()

# sns.lineplot(x = "timepoint", 
#     y="signal", 
#     hue ="event", 
#     style="event",
#     data = df,
#     markers=True,dashes=False,
# )
# plt.show()

# sns.lineplot(x = "timepoint", y="signal", hue ="region", style="event",data = df)
# plt.show()



















# #Simple time series grafic
# import pandas_datareader as pr
# import matplotlib.pyplot as plt
# import pandas as pd

# df = pr.get_data_yahoo("AAPL", start="2016-01-01", end = "2019-08-25")
# # print(df)
# # print(df.shape)

# close = df["Close"]
# close.plot()
# plt.show()

# close.index = pd.DatetimeIndex(close.index)
# print(close.head())