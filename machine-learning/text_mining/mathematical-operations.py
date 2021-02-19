import pandas as pd

text = """
A Scandal in Bohemia! 01
The Red-headed League, 2
The Boscombe Valley Mystery 4
The Man with? the Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Five Orange Pips 1
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""

v_text = text.split("\n")
v = pd.Series(v_text)

v_vector = v[1:len(v)]

mdf = pd.DataFrame(v_vector, columns=["books"])

d_mdf = mdf.copy()

lower_upper_df = d_mdf["books"].apply(lambda x: " ".join(x.lower() for x in x.split() ))
non_punctuation = lower_upper_df.str.replace("[^\w\s]","")
non_numerical = non_punctuation.str.replace("\d","")


d_mdf = pd.DataFrame(non_numerical, columns=["books"])

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

sw = stopwords.words("english")
non_stop_word = d_mdf["books"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


d_mdf = pd.DataFrame(non_stop_word, columns=["books"])

deleted =  pd.Series(" ".join(d_mdf["books"]).split()).value_counts()[-3:]

less_word = d_mdf["books"].apply(lambda x: " ".join(i for i in x.split() if i not in deleted))


from nltk.stem import PorterStemmer

st = PorterStemmer()

stemmed = d_mdf["books"].apply(lambda x: " ".join([st.stem(i) for i in x.split()]))


d_mdf = pd.DataFrame(stemmed, columns=["books"])

d_mdf["counts"] = d_mdf["books"].str.len()
# print(d_mdf)


##Word count

word_count = d_mdf["books"].apply(lambda x: len(str(x).split(" ")))
d_mdf["word_counts"] = word_count
# print(d_mdf)


##Get special Characters and count

d_mdf["adventur_count"] = d_mdf["books"].apply(lambda x: len([x for x in x.split() if x.startswith("adventur")]))
# print(d_mdf)


##Get numerical characters
mdf["numerical_count"] = mdf["books"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(mdf)
