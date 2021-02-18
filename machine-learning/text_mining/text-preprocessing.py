import pandas as pd

text = """
A Scandal in Bohemia! 01
The Red-headed League,2
The Boscombe Valley Mystery4
The Five Orange Pips1
The Man with? the Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""


## Creating Dataframe

# print(text)
# print(text.split())
# print(text.split("\n"))
v_text = text.split("\n")
v = pd.Series(v_text)

v_vector = v[1:len(v)]

mdf = pd.DataFrame(v_vector, columns=["books"])
# print(mdf)


##UpperCase LowerCase

d_mdf = mdf.copy()

lower_upper_df = d_mdf["books"].apply(lambda x: " ".join(x.lower() for x in x.split() ))
# print(lower_upper_df)



##Removing punctuations

non_punctuation = lower_upper_df.str.replace("[^\w\s]","")
# print(non_punctuation)



##Removing numbers

non_numerical = non_punctuation.str.replace("\d","")
# print(non_numerical)



##Removing Stopwords

d_mdf = pd.DataFrame(non_numerical, columns=["books"])

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

sw = stopwords.words("english")
# print(sw)
non_stop_word = d_mdf["books"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
# print(non_stop_word)



## Remove less used words


d_mdf = pd.DataFrame(non_stop_word, columns=["books"])

# print(pd.Series(" ".join(d_mdf["books"]).split()).value_counts())

##we want to delete less frequently words for example last 3
deleted =  pd.Series(" ".join(d_mdf["books"]).split()).value_counts()[-3:]

less_word = d_mdf["books"].apply(lambda x: " ".join(i for i in x.split() if i not in deleted))
# print(less_word)


##Tokenization


nltk.download("punkt")

import textblob

from textblob import TextBlob,Word

d_mdf = pd.DataFrame(less_word, columns=["books"])

# print(TextBlob(d_mdf["books"][1]).words)
tokenized = d_mdf["books"].apply(lambda x: TextBlob(x).words)
# print(tokenized)



##Stemming


from nltk.stem import PorterStemmer

st = PorterStemmer()

# d_mdf = pd.DataFrame(tokenized, columns=["books"])

stemmed = d_mdf["books"].apply(lambda x: " ".join([st.stem(i) for i in x.split()]))
# print(stemmed)



##Lemmatization


nltk.download("wordnet")

# d_mdf = pd.DataFrame(stemmed, columns="books")

lemminized = d_mdf["books"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# print(lemminized)

print(mdf["books"][0:5])
print(d_mdf["books"][0:5])