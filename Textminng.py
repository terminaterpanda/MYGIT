import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import requests
from bs4 import BeautifulSoup

#예시 file -1
file_path1 = "/Users/iseong-yong/Desktop/files/movie1.csv" 
textfile = pd.read_csv(file_path1, encoding= "utf-8")
"""
with open(file_path1, "r) as file:
    textfile = file.read() <when not using pandas library>
"""
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

            with open(filename, "w", encoding = "utf-8") as file:
                file.write(text)
            
            print(f"Saved{filename}")

        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

"""사용법"""

#headers = (useragent name)

textfile_1 = textfile[textfile.point < 5] #580*3 matrix
textfile_2 = textfile[~textfile.index.isin(textfile_1.index)] #420*3 matrix

t1 = textfile_1.iloc[0:580,2].reset_index(drop=True)
t2 = textfile_2.iloc[0:420,2].reset_index(drop=True)
#rest_index == 인덱스를 재정렬해주는 function.

okt = Okt()
texter = " ".join(t1.astype(str))
texter2 = " ".join(t2.astype(str))

nouns = okt.nouns(texter)
nouns1 = okt.nouns(texter2)

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

#지금 이건 빈도수 분석을 한거고, 내가 필요한 건 one-hot encoding된 것을 써야 되는 거 아녀?

print(nouns)  #nouns -> vectorize finished한 negative data
print(nouns1) #nouns1 -> vectorize finished한 positive data

