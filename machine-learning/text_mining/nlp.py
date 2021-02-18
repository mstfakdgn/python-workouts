from textblob import TextBlob
import pandas as pd

##n gram

a = """ Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim. N-gramlar birlikte kullanılan kelimelerin kombinasyonunu gösterir"""
# print(TextBlob(a).ngrams(1))
# print(TextBlob(a).ngrams(2))
# print(TextBlob(a).ngrams(3))

##pos part of speech tagging

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

v_text = text.split("\n")
v = pd.Series(v_text)

v_vector = v[1:len(v)]

mdf = pd.DataFrame(v_vector, columns=["books"])

import nltk

nltk.download("averaged_perceptron_tagger")
d_mdf = mdf["books"].apply(lambda x: TextBlob(x).tags)
# print(d_mdf)


# ##Chunkings (shallow parsing)


# pos = mdf["books"].apply(lambda x: TextBlob(x).tags)

# sentence =" R and Python are useful data science tools for the new or old data scientists who eager to do efficent data science task"
# pos = TextBlob(sentence).tags
# # print(pos)
# reg_exp = "NP: {<DT>?<JJ>*<NN>}"
# rp = nltk.RegexpParser(reg_exp)

# results = rp.parse(pos)
# print(results)
# results.draw()



#Name entity recognition


from nltk import word_tokenize, pos_tag, ne_chunk

nltk.download("maxent_ne_chunker")
nltk.download("words")

sentence2 = "Hadley is creative people who work for R Studio AND he attented conference at Newyork last year"
print(ne_chunk(pos_tag(word_tokenize(sentence2))))