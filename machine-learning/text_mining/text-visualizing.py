import pandas as pd

data = pd.read_csv("../../reading_data/train.tsv", sep = "\t")

# print(data.head())
# print(data.info())

##Upper-Lower case 

data['Phrase'].apply(lambda x: " ".join(x.lower() for x in x.split() ))
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x.lower() for x in x.split()))

##Punctiations

data['Phrase'].str.replace('[^\w\s]','')
data['Phrase'] = data['Phrase'].str.replace('[^\w\s]','')

##Numerical

data['Phrase'].str.replace('\d','')
data['Phrase'] = data['Phrase'].str.replace('\d', '')

## Stopwords

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words("english")
data['Phrase'] = data["Phrase"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

##Less frequent words

deleted =  pd.Series(" ".join(data['Phrase']).split()).value_counts()[-1000:]
data['Phase'] = data['Phrase'].apply(lambda x: " ".join(i for i in x.split() if i not in deleted))

##Lemmatization

from textblob import TextBlob,Word
nltk.download("wordnet")
data['Phase'] = data['Phase'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# print(data['Phase'].head(10))


# ##Terim frequency

# tf1 = (data['Phrase']).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
# tf1.columns=["words", "tf"]

# a = tf1[tf1["tf"] >1000]

# a.plot.bar(x = "words", y = "tf")


##Wordcloud

import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
text = data['Phrase'][0]

wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size = 50, max_words = 100, background_color = "white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

# wordcloud.to_file("wordcloud.png")


## all

text = " ".join(i for i in data.Phrase)

wordcloud = WordCloud(max_font_size = 50, background_color = "white").generate(text)
plt.figure(figsize = [10,10])
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")
