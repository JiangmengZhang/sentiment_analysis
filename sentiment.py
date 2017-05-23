#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pyprind
import pandas as pd
import os
import re
os.chdir('/Users/JiangmengZhang/Documents/Asentimentalproj')
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()

# delete 10008_8.txt in test/pos
# Part I, obtain IMDb movie review dataset
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './aclImdb/%s/%s' % (s,l)
        for file in os.listdir(path):
            if 'txt' in file:
                with open(os.path.join(path, file), 'r', encoding='utf-8') as infile: txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
df.columns = ['review', 'sentiment']

#shuffle            
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
df = pd.read_csv('./movie_data.csv')
df.head(3)    


#Part II: cleaning text data + stop word removal
#import nltk
#nltk.download()
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                 + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# stemming
#from nltk.stem.porter import PorterStemmer
#porter =  PorterStemmer()
#def tokenizer_porter(text):
#    return [porter.stem(word) for word in text.split()]

# Part III: online algorithms and out-of-core learning
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')
   
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
   
X_test, y_test = get_minibatch(doc_stream, size=1000)
X_test = vect.transform(X_test)
print('Accurary: %.3f' % clf.score(X_test, y_test))
clf = clf.partial_fit(X_test, y_test)




