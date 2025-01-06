#NNLM
import pandas as pd
import matplotlib as plt
import urllib.request
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import numpy as np
import random
#data download.
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')
"""
print(train_data[:5])
print(len(train_data))
"""
print(train_data.isnull().values.any())
train_data = train_data.dropna(how = "any")

train_data['document'] = train_data['document'].str.replace('[^가-힣]',"",
                                                            regex=True)
train_data[:5]

okt = Okt()
stopwords = ["가", "밥", "과","도"]
tokenized_data = []
for sentence in train_data(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)
    
class glove():
    def __init__(self, *args):
        self.args = args
        if not args:
            raise ValueError
r1 = re.compile("[^가-힣]")
r2 = re.compile("[^0-9]")
r3 = re.compile("[^a-z]")
r4 = re.compile("[a-]")

vocab_size = 20000
output_dim = 128
input_length = 500
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

X_encoded = tokenizer.texts_to_sequences(sentences)
print("정수 인코딩 결과 :", X_encoded)

max_len = max(len(l) for l in X_encoded)
print(max_len)

X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)
print('패딩 결과 :')
print(X_train)

import numpy as np
class tf_idf():
    def __init__(self, args, value):
        self.args = args
        if not args:
            raise ValueError("error imported")
        self.value = value
        
    def calculate(self, tf):
        self.tf = tf
        a = np.log(self.value / 1+self.args)
        return tf/a

    
        