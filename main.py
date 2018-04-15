import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk
import codecs
import spacy
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.internals import find_jars_within_path
from sklearn import svm
import string
import pickle
from nltk.corpus import stopwords

def text_clean(corpus, keep_list):
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                p1 = p1.lower()
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus


f_train = open("../input/LabelledData (1).txt", 'r')

train = pd.DataFrame(f_train.readlines(), columns = ['Question'])
train.shape

train.iloc[0]

train.iloc[0]
train['QType'] = train.Question.apply(lambda x: (x.split(',')[-1]).strip())
#train['QType'] = train.QType.apply(lambda x: x[:-1])

train.QType.value_counts()

corpus = pd.Series(train.Question.tolist()).astype(str)

common_dot_words = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']
corpus = text_clean(corpus,common_dot_words)
lem = WordNetLemmatizer()
corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
stemmer = SnowballStemmer(language = 'english')
corpus = [[stemmer.stem(x) for x in x] for x in corpus]
corpus = [''.join(x) for x in corpus]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus, train.QType, test_size=0.2, random_state=42)

nlp = spacy.load('en')

#count vectors
#vectorizer = CountVectorizer()
#TF-IDF
#vectorizer = TfidfVectorizer()
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = TfidfVectorizer(ngram_range = (1,1))
#vectorizer = HashingVectorizer(n_features=2000)

vectorizer.fit(corpus)

train_vector = vectorizer.transform(X_train)
test_vector = vectorizer.transform(X_test)
print(train_vector.shape)
print(test_vector.shape)

model = svm.LinearSVC()
model.fit(train_vector, y_train)
preds = model.predict(test_vector)
print(accuracy_score(y_test, preds))

with open ('QuePredictor_train.pkl','wb') as fdump:
    pickle.dump((vectorizer, model), fdump)
    
