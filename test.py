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

with open ('QuePredictor_train.pkl','rb')as f:
   vectorizer,model = pickle.load(f)

new_data = pd.read_csv("test_Question.csv",error_bad_lines=False)

X_test = new_data["Question"]

corpus = pd.Series(X_test.Question.tolist()).astype(str)

common_dot_words = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']
corpus = text_clean(corpus,common_dot_words)
lem = WordNetLemmatizer()
corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
stemmer = SnowballStemmer(language = 'english')
corpus = [[stemmer.stem(x) for x in x] for x in corpus]
corpus = [''.join(x) for x in corpus]


vectored =  vectorizer.transform(corpus)
predicted_label = model.predict(vectored)

new_dict = {

"Question": X_test.Question,     
"Predicted_QType" : predicted_label
     
 }


new_dict_df = pd.DataFrame(new_dict)
new_dict_df.to_csv("results_new.csv", sep=',')
