#korean preprocessing
sent = "전희원님이 개발한 PyKoSpacing은 띄어쓰기가 되어있지 않은 문장을 띄어쓰기를 한 문장으로 변환해주는 패키지입니다. PyKoSpacing은 대용량 코퍼스를 학습하여 만들어진 띄어쓰기 딥 러닝 모델로 준수한 성능을 가지고 있습니다."
new_sent = sent.replace(" ",'')
print(new_sent)

#spacing -> 한국어 띄어쓰기 LLM
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")
#example data.
"""
soynlp -> 학습에 기반한 토크나이저.(학습에 needed 한국어 문서 download.)
"""
corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus)
i = 0
for document in corpus:
    if len(document) > 0:
        print(document)
        i = i+1
    if i == 3:
        break    
word_extractor = WordExtractor()
word_extractor.train(corpus)
word_score_table = word_extractor.extract()
#train해가지고 새로운 단어가 등장할때 등장 빈도룰 바탕으로 tokenize

#이런 식으로 응집도를 계산하여 응집도가 제일 높은 걸 뽑아서 그 단어를 토큰화한다.
#right branching entropy
word_score_table["디스"].right_branching_entropy
word_score_table["디스플레이"].right_branching_entropy

#soynlp tokenizer(L 토큰 + R 토큰의 format.)
from soynlp.tokenizer import LTokenizer

scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
l_tokenizer = LTokenizer(scores=scores)
l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False)

#soynlp를 이용한 반복 문자 정제
from soynlp.normalizer import *
print(emoticon_normalize('앜ㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠ', num_repeats=2))
print(emoticon_normalize('앜ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ이영화존잼쓰ㅠㅠㅠㅠㅠㅠㅠㅠ', num_repeats=2))
#num_repeats를 use하여 반복하는 것들 2회반복으로 줄임.

#사용자 사전 추가하기(단어를 토큰화하는 방법을 알려주는 사전)
from konlpy.tag import Twitter
twitter = Twitter()
twitter.morphs("은경이는 사무실로 갔습니다.")
twitter.add_dictionary("은경이", "Noun")
twitter.morphs("은경이는 사무실로 갔습니다.")
#언어 model -> 단어 시퀸스(문장)에 확률을 할당하는 model
#probs using language model
#SLM
#n-gram
"""
1. 희소문제
trade-off 문제 
2. 훈련 (fine-tuning) data를 무엇으로 보는지에 따라서 difference 생김
"""

#BOW 이해
from konlpy.tag import Okt

okt = Okt()

def build_bag_of_words(document):
    document = document.replace(".", '')
    tokenized_doc = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_doc:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            bow.insert(len(word_to_index) -1, 1)
        else:

            index = word_to_index.get(word)
            bow[index] = bow[index] + 1

    return word_to_index, bow

doc1 = "나는 밥을 먹는다"
vocab, bow = build_bag_of_words(doc1)
#bow -> 단어들의 출현빈도를 indexing하여 계산하는 code

from sklearn.feature_extraction.text import CountVectorizer
corpus1 = ["you know what i love. love."]
vector = CountVectorizer()

print('bag of words vector :', vector.fit_transform(corpus1).toarray())

print(vector.vocabulary_)
#하지만 countvectorizer은 단지 띄어쓰기를 기준으로만 단어 자르는 낮은 수준의 tokenize.

#TF-IDF 기법
#tf-idf를 자동계산해주는 Tfidvectorizer를 use.
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)
#cosine similarity

import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a, b)/norm(a)*norm(b)

#cosine similarity 공식

#vector matrix 연산.

import numpy as np
d = np.array(5)
#스칼라 -> 하나의 실수값으로 이루어진 tensor.