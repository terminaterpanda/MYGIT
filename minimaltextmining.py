import pandas as pd
import numpy as np
import torch 
# 딥 러닝 용 package.
from sklearn.model_selection import train_test_split
# 학습 data, 그리고 시험용 data 분리하는 package.
from sklearn.metrics import accuracy_score
#정확도 점수 계산
from sklearn.svm import SVC
#svc = 커널용 package
from sklearn.preprocessing import LabelEncoder
import requests
#data scrap-ing 할때(requests)응답
import re
#정규식 import 
import time
#시간 import - data name 따로 지정하기 위해서 use.
from bs4 import BeautifulSoup
#data scraping 하는 package
from datetime import datetime
from transformers import AutoTokenizer, AutoModel



def clean_text(text):
    return re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)).strip()

class Scraping:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        #scraping에 stack 형태로 넘겨줌

    def scrap_save(self):
        try:
            res = requests.get(self.url, headers=self.headers)
            res.raise_for_status()  # status를 정의하여 가지고 올 수 있는지를 확인.

            soup = BeautifulSoup(res.text, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            # \n을 사용해서 줄바꿈으로 seperator use.

            sentences = text.split(".")  # 구두점으로 sentence 분리.
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

            korean_sentences = [
                sentence for sentence in sentences if re.search(r"[가-힣]", sentence)
            ]
            df = pd.DataFrame(korean_sentences, columns=["sentence"])

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #timestamp -> 시간을 측정해서 file 이름을 넣고, 그런 상태로 filename.csv 로 저장.
            filename = f"korean_sentences_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"saved {filename}")
            self.df = df
            return df
        
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            return pd.DataFrame()
        except Exception as err:
            print(f"An error occurred: {err}")
            return pd.DataFrame()

    def continuous_scrap(self, refresh_interval, min_data_count = 200):
        
        collected_data = pd.DataFrame()
        while True:
            print("refresh")
            new_data = self.scrap_save()
            collected_data = pd.concat([collected_data, new_data]).drop_duplicates().reset_index(drop = True )

            if len(collected_data) >= min_data_count:
                print("not yet")
                return collected_data
            print(f"{refresh_interval} 동안 대기")
            time.sleep(refresh_interval)


class GET():
    def __init__(self, name, file_path):
        self.name = name
        self.file_path = file_path
        textfile = pd.read_csv(file_path, encoding ="utf-8")
        textfile = textfile.replace(".", "")
        self.textfile = textfile

class SentenceEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
        #finetuning gogo
        self.model = AutoModel.from_pretrained("skt/kobert-base-v1")
        self.device = torch.device("mps" if torch.has_mps else "cpu")
        self.model.to(self.device) #mps torch를 use해서 GPU use.

    def get_embedding(self, sentence):
        try:
            if pd.isna(sentence) or not sentence.strip():
                return np.zeros(768)
        #np.zeros 0vector로 변환 -> NA값들을))
            inputs = self.tokenizer(sentence, return_tensors="pt",padding="max_length", truncation=True, max_length=128
                                )
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
        #inputs는 tokenizer 생성된 dict. 여기에서 이 키- 값 쌍을 전부 mps.self.device에 옮김
            with torch.no_grad(): #no_grad -> 자동 미분(gradient 비활성화)
                outputs = self.model(**inputs)
                cls_embedd = outputs.last_hidden_state[:, 0, :]
                return cls_embedd.squeeze().cpu().numpy()
        except Exception as e:
            print(f"warning: '{sentence}' - {e}")
            return np.zeros(768)
        
class SentimentalModel:
    def __init__(self):
        self.embedding_model = SentenceEmbedding()
        self.vectorizer = None
        self.model = SVC(kernel="linear") #초기화하여 모델 학습에 use
        #linear 선형 model을 use.
        self.label_encoder = LabelEncoder()
        #labelencoder = textlabel을 정수로 encoding + decoding.

    def train_model(self, df_neg, df_pos):
        df_neg["label"] = 0
        df_pos["label"] = 1
        data = pd.concat([df_neg, df_pos], ignore_index= True)

        data['embedding'] = data['sentence'].apply(lambda x: self.embedding_model.get_embedding(x).flatten())
        x = np.vstack(data["embedding"].values)
        y = self.label_encoder.fit_transform(data["label"].values)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"accuracy : {accuracy * 100:.2f}%") #퍼센트age. 

    def predict_sentiment(self, sentence):
        embedding = self.embedding_model.get_embedding(sentence).flatten()
        if embedding is not None and len(embedding) == 768:
            prediction = self.model.predict([embedding])
            return self.label_encoder.inverse_transform(prediction)[0]
        return "Unknown"


url = "https://news.naver.com/"
headers = {"User-Agent": 
           "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}

#headers는 키-값(value) 형식으로 저장되어야 한다.

scraper = Scraping(url, headers)
sentiment_model = SentimentalModel()

refresh_interval = 600

while True:
    print("refreshing data...")
    new_data = scraper.scrap_save()

    if not new_data.empty:
        new_data["sentence"] = new_data["sentence"].apply(clean_text)
        for sentence in new_data['sentence']:
            sentiment = sentiment_model.predict_sentiment(sentence)

            print(f"sentence : {sentence}" | "predict : {sentiment}")

    time.sleep(refresh_interval)