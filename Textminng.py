import numpy as np
import re
import pandas as pd
from konlpy.tag import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt

#기본적인 값을 가지고 올 수 있는 함수 정의.

class Scraping:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers

    def scrap_save(self, filename):
        try:
            res = requests.get(self.url, headers = self.headers)
            res.raise_for_status() #status를 정의하여 가지고 올 수 있는지를 확인.

            soup = BeautifulSoup(res.text, "lxml")
            text = soup.get_text(separator='\n', strip=True)
            #\n을 사용해서 줄바꿈으로 seperator use.

            sentences = text.split(".") #구두점으로 sentence 분리. 
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

            korean_sentences = [sentence for sentence in sentences if re.search(r"가-힣", sentence)]
            df = pd.DataFrame(korean_sentences, columns=["sentence"])
            df.to_csv(filename, index = False, encoding="utf-8-sig")
            print(f"saved {filename}")

        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

"""사용법""" #scraping으로 get, 전처리 후 아래의 사용법으로 use.
############################################## scraping finish.

#예시 file -1
file_path1 = "/Users/iseong-yong/Desktop/files/movie1.csv" 
textfile = pd.read_csv(file_path1, encoding= "utf-8")
"""
with open(file_path1, "r) as file:
    textfile = file.read() <when not using pandas library>
"""
#headers = (useragent name)

textfile = textfile.replace(".", '')

# text를 평점에 따라서 구분정의.
textfile_1 = textfile[textfile.point < 5] #580*3 matrix
textfile_2 = textfile[~textfile.index.isin(textfile_1.index)] #420*3 matrix

t1 = textfile_1.iloc[:,2].reset_index(drop=True)
t2 = textfile_2.iloc[:,2].reset_index(drop=True)
#rest_index == 인덱스를 재정렬해주는 function.

okt = Okt()
texter = " ".join(t1.astype(str))
texter2 = " ".join(t2.astype(str))

nouns = okt.morphs(texter)
nouns1 = okt.morphs(texter2)

#texter == "형태소 분류 finish."

#명사 빈도수 계산
nouns_freq = pd.Series(nouns).value_counts()
nouns_freq1 = pd.Series(nouns1).value_counts()
#value_counts == 기본적으로 지정된 열의 값들에 대한 모든 밞생 횟수를 반환.

nouns_df = nouns_freq.reset_index()
nouns_df2 = nouns_freq1.reset_index()

nouns_df.columns = ["noun", "frequency"]
nouns_df2.columns = ["noun", "frequency"]

print(nouns_df)
print(nouns_df2)

output_file_path = "/Users/iseong-yong/Desktop/files/nouns_frequency_negative.csv"
output_file_path1 = "/Users/iseong-yong/Desktop/files/nouns_frequency_positive.csv"

nouns_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
nouns_df2.to_csv(output_file_path1, index=False, encoding="utf-8-sig")
#nouns_df == 평점 5점 미만
#nouns_df2 == 평점 5 이상 (vector 임베딩)

stopwords = ['이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같', '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하']

filteredn = nouns_df[~nouns_df["noun"].isin(stopwords)]
filteredp = nouns_df2[~nouns_df2["noun"].isin(stopwords)]
#match == "re에서 사용" 다른 곳에서는 isin use.

file_path3 = "/Users/iseong-yong/Desktop/files/negative.csv"
file_path4 = "/Users/iseong-yong/Desktop/files/positive.csv"

filteredn.to_csv(file_path3, index=False, encoding="utf-8-sig")
filteredp.to_csv(file_path4, index=False, encoding="utf-8-sig")

print(filteredn)
print(filteredp)

#--------------------------------------------------------------(단어(noun)출현 빈도 분석) <pos vs neg>
################################################################

#감정 분석 start.

texter3 = texter.split(".")
texter4 = texter2.split(".")

#texter3 = "온점을 기준으로 문장을 쪼갬" (negative)
#texter4 = "온점을 기준으로 문장을 쪼갬" (positive)

def morph_sentance(sentences):
    result = []
    for sentence in sentences:
        if sentence.strip():
            result.append(" ".join(okt.morphs(sentence.strip())))
    return result #for loop finish 되고 나서 return 문 실행.

texter3 = morph_sentance(texter3)
texter4 = morph_sentance(texter4)

print(texter3) #texter3 == negative data(온점으로 쪼개고, 형태소분석을 완료한것)
print(texter4) #texter4 == positive data(온점으로 쪼개고, 형태소분석을 완료한것)

#정규식 정의
def clean_text(text):
    text = re.sub("[.]+", " ", text)
    text = re.sub("[,]+", " ", text)
    text = re.sub("[_]+", " ", text)
    text = re.sub(r'[^가-힣\s]', '', text)
    text = text.strip()
    return text
"""
a4 = re.compile()
a5 = re.compile()
a6 = re.compile()
"""
#한글-> [가-힣](장규표현식)

cleaned1 = [clean_text(sentence) for sentence in texter3]
cleaned2 = [clean_text(sentence) for sentence in texter4]

df_neg = pd.DataFrame(cleaned1, columns=["sentence"])
df_pos = pd.DataFrame(cleaned2, columns=["sentence"])

file_path5 = "/Users/iseong-yong/Desktop/files/neg.csv"
file_path6 = "/Users/iseong-yong/Desktop/files/pos.csv"

df_neg.to_csv(file_path5, index=False, encoding="utf-8-sig")
df_pos.to_csv(file_path6, index=False, encoding="utf-8-sig")

#neg, pos 저장(in csv)
X_neg = df_neg["sentence"]
X_pos = df_pos["sentence"]

y_neg = [0] * len(X_neg)
y_pos = [1] * len(X_pos)

X = pd.concat([X_neg, X_pos], ignore_index=True)
y = y_neg + y_pos

vectorizer = CountVectorizer() 
X_Vectorized = vectorizer.fit_transform(X)
# 수치형 data로 입력받아야 하므로 문자열 data를 수치형 배열로 변환시키고 삽입해야 한다.
X_train, X_test, y_train, y_test = train_test_split(X_Vectorized, y, test_size=0.2, random_state=28)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

###########################################################

def predict_sentiment(sentence):
    sentence = " ".join(okt.morphs(sentence))
    vectorized_sentence = vectorizer.transform([sentence])
    return model.predict(vectorized_sentence)[0]
#bow 형태를 활용하여 수치화한 data이므로, 단어의 빈도를 계산하여 벡터화하기 때문애 새로운 data에는 적응할 수 없다.

ex0 = "오래도록 기억 에 남을것 같아요"
result = predict_sentiment(ex0)
print(f"sentiment : {result}")

###################################################
# Code Implementation Usng Bert

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# load pre-trained bert model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModel.from_pretrained("skt/kobert-base-v1")
#pre-trained model을 가져오기.

def get_sentence_embedding(sentence):
    try:
        if pd.isna(sentence) or not isinstance(sentence, str) or sentence.strip() == "":
            # Handle invalid or empty sentences
            return np.zeros(768)
        
        inputs = tokenizer(sentence, return_tensors="pt", padding='max_length',
                           truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.detach().numpy()  # Convert to NumPy array
        
    except Exception as e:
        print(f"Error processing sentence: '{sentence}' - {e}")
        return np.zeros(768)  # Return zero vector for error cases

data1 = pd.read_csv(file_path5, encoding = "utf-8") #neg data
data2 = pd.read_csv(file_path6, encoding = "utf-8") #pos data

data1['embedding'] = data1["sentence"].apply(lambda x: get_sentence_embedding(x).flatten() if x is not None else np.zeros(768))
data2['embedding'] = data2["sentence"].apply(lambda x: get_sentence_embedding(x).flatten() if x is not None else np.zeros(768))

data2['label'] = 0 #data2 = negative(label = 0)
data1['label'] = 1 #data1 = positive(label = 1)
#해당 bert embedding으로 변환.(1d 배열로 변환) = embedding 이라는 새로운 열에 저장.

data1 = pd.concat([data1, data2], ignore_index=True)
#data를 붙이고, index를 삭제하여 순서만 매김.

x = np.vstack(data1['embedding'].values)
#np.vstack 즉, 그냥 vertical로 합쳐버리는 함수이다.
y = data1['label'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#train-test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=98)

clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def predict_sentiment(sentence):
    embedding = get_sentence_embedding(sentence).flatten()
    
    if embedding is not None and len(embedding) == 768:
        prediction = clf.predict([embedding])
        return label_encoder.inverse_transform(prediction)[0]
    else:
        return "Unknown"  # Or return some default sentiment

test_sentence = "기생충은 정말 잘못됬다."
result = predict_sentiment(test_sentence)

print(f"predicted sentiment: {result}")