import pandas as pd
import numpy as np
import requests
import time
from bs4 import BeautifulSoup

result = pd.DataFrame() #data 저장할 빈 dataframe 생성
for i in range(584274, 595226): #for문을 지정하여 data를 갖고 오기
    URL = "http://www1.president.go.kr/petitions/"+str(i)
    
    response = requests.get(URL) #get.URL으로 응답을 response에 저장.
    html = response.text #그 응답중 text만 뽑아서 html로 저장하기
    soup = BeautifulSoup(html, "html.parser") #beautifulsoup으로 data 가공
    title = soup.find("h3", class_ = 'petitionsView_title') #위치 속성값을 뽑아서 가져오기
    count = soup.find('span', class_ = 'counter') #참여인원 속성값도 가져오기
    
    for content in soup.select("div.petitionsView_write > div.View_write"):
        content
    a = []
    for tag in soup.select('ul.petitionsView_info_list > li'):
        a.append(tag.contents[1])
    if len(a) != 0:
        df1 = pd.DataFrame({"start" : [a[1]],
                            "end" : [a[2]],
                            "category" : [a[0]],
                            'count' : [count.text],
                            'title' : [title.text],
                            'content' : [content.text.strip()[0: 13000]]
                            }) #content.text.strip()으로 공백을 제거하고 데이터의 길이를 13000으로 제한.
        result=pd.concat([result,df1])
        result.index=np.arange(len(result))
    if i % 60 == 0: #60건의 글을 크롤링할때마다 90초를 멈춘 후 재작업.
        print("sleep 90seconds. Count:" + str(i)
              +", Local Time:"
              + time.strftime('%Y-%m-%d', time.localtime(time.time()))
              +" "+time.strftime('%X', time.localtime(time.time()))
              +",  Data Length:"+str(len(result)))
        time.sleep(90)
        df = result
        
print(result.shape)
df = result
df.head()

import re

def remove_white_space(text):
    text = re.sub(r'[\t\r\n\f\v], " ', str(text))
    return text

def remove_special_chr(text):
    text = re.sub("[^ ㄱ-|가-힣 0-9]+", " ", str(text))
    return text

df.title = df.title.apply(remove_white_space)
df.title = df.title.apply(remove_special_chr)

df.content = df.content.apply(remove_white_space)
df.content = df.content.apply(remove_special_chr)

from konlpy.tag import Okt

okt = Okt()

df["title_token"] = df.title.apply(okt.morphs)
df["content_token"] = df.content.apply(okt.nouns)

df["token_final"] = df.title_token + df.content_token

df["count"] = df["count"].replace({"," : ""}, regex=True).apply(lambda x : int(x))

print(df.dtypes)

df['label'] = df['count'].apply(lambda x: "Yes" if x>1000 else "No")

df_drop = df[["token_final", "label"]]

from gensim.models import Word2Vec
#gensim models == "자연어처리에 사용되는 model들을 지정
# 
embedding_model = Word2Vec(df_drop["token_final"], #token_final == 임베딩 벡터를 생성할 대상 data
                           sg = 1, #model 구조 옵션지정(sg = 1-> skip_gram/ sg = 0 -> cbow
                           size = 100, #임베딩 벡터의 크기를 지정
                           window = 2, #윈도우 지정
                           min_count=1,#일정 횟수 이상 등장하지 않는 토큰을 임베딩 벡터에서 제외.
                           
                           workers=4) #workers -> 실행할 병렬 프로세서의 수(4~6)의 값을 지정.

print(embedding_model)

model_result = embedding_model.wv.most_similar("음주운전")
print(model_result)

#embedding model 저장 및 load

from gensim.models import KeyedVectors
#embedding model 불러오기 위한 클래스

embedding_model.wv.save_word2vec_format("data/petitions_tokens_w2v")
#임베딩 모델을 로컬 data folder에서 이름으로 저장
loaded_model = KeyedVectors.load_word2vec_format("data/petitions_tokens_w2v")
#그 모델을 loaded_model에 저장
model_result = loaded_model.most_similar("음주운전")
print(model_result)

from numpy.random import RandomState

rng = RandomState()

tr = df_drop.sample(frac=0.8, random_state=rng)
#0.8frac(총 data의 80%를 가져오기)
val = df_drop.loc[~df_drop.index.isin(tr.index)]
#나머지 20%를 validation set으로 지정
tr.to_csv("data/train.csv", index=False, encoding="utf-8-sig")
val.to_csv("data/validation.csv", index=False, encoding="utf-8-sig")

import torchtext
from torchtext.data import Field

def tokenizer(text):
    text = re.sub("[\[\]\"]", "", str(text)) #token_final에서 제거하여 변경
    text = text.split(', ') #위 문자열을 구분자로 분리한 결과를 반환
    return text

TEXT=Field(tokenize = tokenizer)
LABEL=Field(seqential = False)

import torch
from torchtext.vocab import Vectors
