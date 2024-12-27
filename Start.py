import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt")
nltk.download("stopwords")

#numpy = (1.ndim == 축의 개수) / (2.shape == 크기)

zer0_mat = np.zeros((6,6)) # 모든 원소 : 0
print(zer0_mat)

one_mat = np.ones((3,3)) # 모든 원소 : 1
print(one_mat)

full_mat = np.full((3,3), 22) # 모든 원소 : 22(사용자 지정)
print(full_mat)

eye_mat = np.eye(6)
print(eye_mat) # 단위 행렬 생성

random_mat = np.random.random((4,4)) #random한 임의의 값을 가지는 배열 생성
print(random_mat)

n = 3
range_step = np.arange(1, 12, n) #1부터 (12-1)까지 3씩 증가하는 배열 생성
print(range_step)

reshape_mat = np.array(np.arange(40)).reshape((5,8))
print(reshape_mat) #reshape_mat == 원소 변경하지 x, 배열의 구조만 change.

#numpy slicing
mt = np.array([[1,2,3], [1,5,6]])
slice_mat = mt[:,1] #두번째 열[행,열]
print(slice_mat)

print(mt[1,1]) #indexing using numpy 배열


#np.add == + / np.substract == - / np.multiply == * / np.divide == %
#요소별 곱에서만 적용.(행렬곱은 dot use.)

#시각화 (line plot 그리기) = "matplotlib.pyplot use"
plt.title('test')
plt.plot([1,2,3,4], [2,4,8,6])
plt.plot([3,4,5,6], [11,23,45,65])
plt.xlabel('hour')
plt.ylabel('money')
plt.legend(["a", "b"])
plt.show()
#xlabel == x축이름 / ylabel == y축이름

#########################################################(text-embedding)

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize

tokenizer = TreebankWordTokenizer()
tokenizer1 = sent_tokenize

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own." 

print(tokenizer.tokenize(text))

print(sent_tokenize(text))

from nltk.tag import pos_tag
text1 = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
token_sentence = word_tokenize(text1)

print(token_sentence)
print(pos_tag(token_sentence))

from konlpy.tag import Okt  #open korean text
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 
#morphs == 형태소 추출
#pos == 품사 태깅
#nouns == 명사 추출

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# -> 표제어 추출하는 도구

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print(words)
print([lemmatizer.lemmatize(word) for word in words])

print([lemmatizer.lemmatize("dies", "v")])
#구체적으로 insert한 경우 정확도 상승

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
tokenized_sentence = word_tokenize(sentence)

print(tokenized_sentence)
print([stemmer.stem(word) for word in tokenized_sentence])
#porterstemmer은 규칙기반으로 단어를 정제.

from nltk.stem import LancasterStemmer

lancaster = LancasterStemmer()
words1 = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print(words)

print([stemmer.stem(w) for w in words1])
print([lancaster.stem(w) for w in words1])

#stopwords 지정

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

stop_words_list = stopwords.words('english')
print(len(stop_words_list))

example = "Family is not an important thing. It's everything."
word_tokens = word_tokenize(example)
result = []
for word in word_tokens:
    if word not in stop_words_list:
        result.append(word)

#정규 표현식

import re

r = re.compile("a.v")
r.search("erre")

r1 = re.compile("ab?c")
r1.search("abbbbg")

r2 = re.compile("ac*d")
r2.search("acccd")

r3 = re.compile("aaa+d")
r3.search("aaad")

r4 = re.compile("^ddf")
r4.search("dddf")
#^ 시작 문자열 대표

r5 = re.compile("av{3}d")
r5.search("avvvd")
# v*3개가 존재하는 문자열

r6 = re.compile("add{5,7}c")
r6.search("addd")
#5이상 7이하 
#{4,}-> 4번 이상 repeat
#[]-> 내부의 문자들 중 한개의 문자와는 매치.
#[^] -> 해당 문자를 제외한 문자를 match.
#역 슬래쉬 문자 자체를 의미하는 문법

#\\d-> 모든 숫자
#\\D 숫자를 제외한 모든 문자
#\\s -> 공백을 의미
#\\w -> 문자 혹은 숫자
#\\W -> 문자 또는 숫자가 아닌 문자를 의미

"""
re.compile - 컴파일러
re.search - 매치되는지 검색
re.match - 문자열의 처음이 매치
re.split - 문자열을분리, 리스트로 리턴
re.findall(모든 경우 문자열을 찾아서 list로 리턴)
re.sub - 대체
"""

#정수 인코딩
#dictionary 사용

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
sentence1 = sent_tokenize(raw_text)

vocab = {}
preprocess_Sentence = []
stop_words = set(stopwords.words('english'))

for sentence in sentence1:
    tokenize_sen = word_tokenize(sentence)
    #sentence1 리스트에서 sentence를 뽑아내야함.(sentence1은 리스트이므로 insert하면 오류)
    result = []

    for word in tokenize_sen:
        word = word.lower()

        if word not in stop_words:
            if len(word) > 2:

                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1

    preprocess_Sentence.append(result)
print(preprocess_Sentence)
print(vocab)
print(vocab, ["barber"])

vocab_sorted = sorted(vocab.items(), key= lambda x:x[1], reverse=True)
print(vocab_sorted)

word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted :
    if frequency > 1:
        i = i + 1
        word_to_index[word] = i

print(word_to_index)

vocab_size = 5
word_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]
for w in word_frequency:
    del word_to_index[w]
print(word_to_index)

#word_to_index -> 빈도 수 높은 상위 5개의 단어만 define
#하지만 이 list에 존재하지 않는 단어들은 OOV(out of vocabulary)라고 일컫는다

word_to_index["OOV"] = len(word_to_index) + 1
print(word_to_index)
#dataframe에서의 열 하나를 추가하여 ["OOV"]로 지정하여 분류

encoded_sentences = []
for sentence in preprocess_Sentence:
    encoded_sentence = []
    for word in sentence:
        try:
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            encoded_sentence.append(word_to_index["OOV"])

    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)       

# using Counter

from collections import Counter

print(preprocess_Sentence)
all_words_list = sum(preprocess_Sentence, [])
print(all_words_list)


vocab = Counter(all_words_list)
print(vocab)
#counter -> 단어의 빈도수를 기록
print(vocab["barber"])

#one-hot encoding

from konlpy.tag import Okt

#1. 정수 인코딩 수행
#2. 1과 0을 부여하여 one-hot vectorize.
okt = Okt()
tokens = okt.morphs("나는 밥을 먹는다")
print(tokens)

word_to_index1 = {word : index for index, word in enumerate(tokens)}
print(word_to_index1)

def one_hot_encoding(word, word_to_index1):
    one_hot_vector = [0]*(len(word_to_index1))
    index = word_to_index1[word]
    one_hot_vector[index] = 1
    return one_hot_vector

one_hot_encoding("자연어", word_to_index1)

from keras_preprocessing import Tokenizer

tokenizer = Tokenizer()
text2 = "위에서는 원-핫 인코딩을 이해하기 위해 파이썬으로 직접 코드를 작성하였지만, 케라스는 원-핫 인코딩을 수행하는 유용한 도구"
tokenizer.fit_on_texts([text2])

print(tokenizer.word_index)
#DTM과 유사.

#data seperate.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#training data + test data
#zip function
"""
zip() -> 동일 개수 (각 순서에 등장하는 원소끼리 묶어줌.)
#1,2,3 순서대로 원소가 등장하는 경우에 첫번째, 두번쨰 이런 식으로 묶어줌.
"""
x, y = zip(["a", 1], ["b", 2], ["c", 3])
print(x)
print(y)

#data x, y -> 데이터프레임에서 열 꺼내올때 use.
file_path = " /Users/iseong-yong/Desktop/files/nouns_frequency_negative.csv "
ad = pd.read_csv(file_path, encoding = "utf-8")
av = ad["noun"]
print(av.to_list)
#list 형태로 make.

np_array = np.arange(0, 16).reshape((4,4))
print(np_array)

#train_test_split혹은 수동으로 data 분리.

print(len(av))

num_train = int(len(av) * 0.8)
num_test = int(len(av) - num_train)

av_train = av[num_train:]
av_test = av[num_test:]
print(len(av_train))
print(len(av_test))

#keras importing + modeling.
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
train_text = "The earth is an awesome place live"

tokenizer.fit_on_texts([train_text])
sub_text = "The earth is an awesome place live"
sequences = tokenizer.texts_to_sequences([sub_text])[0]

print(sequences)
print(tokenizer.word_index)
#pad_sequences -> zero-padding(0을 넣어서 길이 계산 맞춤)

pad_sequences([1,2,3], [3,4,5,6], [7,8], maxlen=3,padding="pre")
#maxlen -> data에 대해서 정규화할 길이
#padding = "pre"(앞에 0을 채움) / "padding = "post"(뒤에 0을 채움)

#Embedding() -> 단어를 밀집 vector(원핫 인코딩 임베딩은 너무 크므로 밀집 vector로 표현)

tokenized_text = [['Hope', 'to', 'see', 'you', 'soon'], ['Nice', 'to', 'see', 'you', 'again']]

encoded_text = [[0, 1, 2, 3, 4],[5,1,2,3,6]]

vocab_size = 7
embedding_dim = 2

import tensorflow as tf

class LinearRegression(tf.keras.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(1, input_dim =1,
                                                  activation = "linear")
    def call(self, x):
        y_pred = self.linear_layer(x)

        return y_pred
    
model = LinearRegression()

#MLP(다층 퍼셉트론)
import numpy as np
from keras_preprocessing.text import Tokenizer

texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)

#texts_to_matrix
print(tokenizer.texts_to_matrix(texts, mode="count"))
#texts를 input으로 넣고 모드는 "count"를 use.

print(tokenizer.texts_to_matrix(texts, mode="binary"))
#해당 단어 존재 여부

print(tokenizer.texts_to_matrix(texts, mode = "tfidf").round(2))
#tf-idf

print(tokenizer.texts_to_matrix(texts, mode = "freq").round(2))
#각 단어 등장 횟수 -> 분자
#각 문서 크기 -> 분모  
# -> 이런 식의 분석 way.

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer
from keras_utils import to_categorical

newsdata = fetch_20newsgroups(subset="train")

print(newsdata.keys())
print("훈련용 샘플의 개수 : {}".format(len(newsdata.data)))

print("총 주제의 개수 : {}".format(len(newsdata.target_names)))
print(newsdata.target_names)

import torch 
print(torch.backends.mps.is_built())
#mps를 use가 가능하다.
print(torch.backends.mps.is_available())
#torch.cuda.is_avaliable()과 동일.

import numpy as np

timesteps = 10
input_dim = 4
hidden_units = 8

inputs = np.random.random((timesteps, input_dim))

hidden_state_t = np.zeros((hidden_units,))

print(hidden_state_t)

Wx = np.random.random((hidden_units, input_dim))
Wh = np.random.random((hidden_units, hidden_units))

b = np.random.random((hidden_units,))

print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))

file_Path1 = " /Users/iseong-yong/Desktop/files/kowiki-20241201-stub-articles.xml"
import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#tensor -> torch()
b = 64
learning_rate = 1e-78
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3, ), (0.7,))
    
])
train_dataset = datasets.MNIST(root="/data", train = True,transform=transform,
                               download=True)
train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, padding=3)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=4, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x
    
mps_device = torch.device("mps")
#기본 setting finish.
