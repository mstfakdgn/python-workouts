from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition,ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text,sequence
from keras import layers, models, optimizers

import pandas as pd

data = pd.read_csv("../../reading_data/train.tsv", sep="\t")

data["Sentiment"].replace(0,value = "negative", inplace=True)
data["Sentiment"].replace(1,value = "negative", inplace=True)

data["Sentiment"].replace(3,value = "positive", inplace=True)
data["Sentiment"].replace(4,value = "positive", inplace=True)

data = data[(data.Sentiment == "negative") | (data.Sentiment == "positive")]
# print(data.head())
# print(data.groupby("Sentiment").count())

df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]


##Preprocessings

##Upper-Lower case 

df["text"].apply(lambda x: " ".join(x.lower() for x in x.split() ))
df["text"] = df["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

##Punctiations

df["text"].str.replace('[^\w\s]','')
df["text"] = df["text"].str.replace('[^\w\s]','')

##Numerical

df["text"].str.replace('\d','')
df["text"] = df["text"].str.replace('\d', '')

## Stopwords

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words("english")
df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

##Less frequent words

deleted =  pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]
df["text"] = df["text"].apply(lambda x: " ".join(i for i in x.split() if i not in deleted))

##Lemmatization

from textblob import TextBlob,Word
nltk.download("wordnet")
df["text"] = df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# print(df.head())





###Variable Engineering
# #.Count Vectors
# #.TF-IDF Vectors (words, characters, n-grams)
# #.Word Embeddings

# #TF(t) = (Bir t teriminin bir dökğmanda gözlenme frekansı ) / ( dökümandaki toplam veri sayısı)
# #IDF(t) = log_e(Toplam döküman sayısı/ içinde t terimi olan belge sayısı)


X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"])

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
# print(y_train[0:5])



##Count Vectors

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

X_train_count = vectorizer.transform(X_train)
X_test_count = vectorizer.transform(X_test)

print(vectorizer.get_feature_names()[0:5])
print(X_train_count.toarray())


##TF-IDF

tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(X_train)

X_train_tf_idf_word = tf_idf_word_vectorizer.transform(X_train)
X_test_tf_idf_word = tf_idf_word_vectorizer.transform(X_test)

print(tf_idf_word_vectorizer.get_feature_names()[0:5])
print(X_train_tf_idf_word.toarray())


#ngram level tf-idf

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizer.fit(X_train)

X_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(X_train)
X_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(X_test)

print(tf_idf_word_vectorizer.get_feature_names()[0:5])
print(X_train_tf_idf_word.toarray())


#chracter level tf-idf

tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(X_train)

X_train_tf_idf_char = tf_idf_chars_vectorizer.transform(X_train)
X_test_tf_idf_char = tf_idf_chars_vectorizer.transform(X_test)

print(tf_idf_word_vectorizer.get_feature_names()[0:5])
print(X_train_tf_idf_word.toarray())



## Classsification Logistic Regression

loj = linear_model.LogisticRegression()
loj_model = loj.fit(X_train_count, y_train)
accuracy = model_selection.cross_val_score(loj_model, X_test_count, y_test, cv=10).mean()

print("Logistic Regression ==> Count Vectors Accuracy Rate", accuracy)


loj = linear_model.LogisticRegression()
loj_model = loj.fit(X_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(loj_model, X_test_tf_idf_word, y_test, cv=10).mean()

print("Logistic Regression ==> Word level TF_IDF Accuracy Rate", accuracy)


loj = linear_model.LogisticRegression()
loj_model = loj.fit(X_train_tf_idf_ngram, y_train)
accuracy = model_selection.cross_val_score(loj_model, X_test_tf_idf_ngram, y_test, cv=10).mean()

print("Logistic Regression ==> Ngram TF_IDF Accuracy Rate", accuracy)

loj = linear_model.LogisticRegression()
loj_model = loj.fit(X_train_tf_idf_char, y_train)
accuracy = model_selection.cross_val_score(loj_model, X_test_tf_idf_char, y_test, cv=10).mean()

print("Logistic Regression ==> Char TF_IDF Accuracy Rate", accuracy)



## Classsification Naive Bayes


nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(X_train_count, y_train)
accuracy = model_selection.cross_val_score(nb_model, X_test_count, y_test, cv=10).mean()

print("Naive Bayes ==> Count Vectors Accuracy Rate", accuracy)


nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(X_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(nb_model, X_test_tf_idf_word, y_test, cv=10).mean()

print("Naive Bayes ==> Word level TF_IDF Accuracy Rate", accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(X_train_tf_idf_ngram, y_train)
accuracy = model_selection.cross_val_score(nb_model, X_test_tf_idf_ngram, y_test, cv=10).mean()

print("Naive Bayes ==> Ngram TF_IDF Accuracy Rate", accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(X_train_tf_idf_char, y_train)
accuracy = model_selection.cross_val_score(nb_model, X_test_tf_idf_char, y_test, cv=10).mean()

print("Naive Bayes ==> Char TF_IDF Accuracy Rate", accuracy)



new_comment = pd.Series("this film is very nice and good i like it")
second_new_comment = pd.Series("no not good film that shit is very bad")
v = CountVectorizer()
v.fit(X_train)
new_comment = v.transform(new_comment)
second_new_comment = v.transform(second_new_comment)
print(loj_model.predict(new_comment))
print(loj_model.predict(second_new_comment))
