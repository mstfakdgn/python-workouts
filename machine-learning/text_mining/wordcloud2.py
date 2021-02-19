import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


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



tree_mask = np.array(Image.open("../../reading_data/trees.png"))

text = " ".join(i for i in data.Phrase)

wc = WordCloud(background_color = "white", max_words=1000, mask=tree_mask, contour_width=3, contour_color="firebrick")
wc.generate(text)

plt.figure(figsize=[10,10])
plt.imshow(wc, interpolation = "bilinear")
plt.axis("off")
plt.show()

wc.to_file("wordcloud-trees.png")
