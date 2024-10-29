import pandas as pd
import numpy as np
import torch as nn
from bs4 import BeautifulSoup
import requests
import re
import time
from datetime import datetime

class clean():

    def __init__(self, text):
        self.text = text

    def clean_text(self):
        return re.sub(r"[^가-힣a-zA-Z\s]", "", str(self.text)) if isinstance(self.text, str) else ""

class Scraping:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers

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
            

        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except Exception as err:
            print(f"An error occurred: {err}")

    def continuous_scrap(self, refresh_interval):


        while True:
            print("refresh")
            self.scrap_save()
            print(f"{refresh_interval} 동안 대기")
            time.sleep(refresh_interval)


class GET():
    def __init__(self, name, file_path):
        self.name = name
        self.file_path = file_path
        textfile = pd.read_csv(file_path, encoding ="utf-8")
        textfile = textfile.replace(".", "")
        self.textfile = textfile

url = "https://news.naver.com/"
headers = {"User-Agent": 
           "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
mps_device = nn.device("mps")
#headers는 키-값(value) 형식으로 저장되어야 한다.

scraper = Scraping(url, headers)


refresh_interval = 600
scraper.continuous_scrap(refresh_interval)
filepath = "/Users/iseong-yong/Desktop/files"
get_instance = GET("korean_sentences", filepath)

print(get_instance.textfile.head())


class Moduler:
    def __init__(self, moduler):
        self.moduler = moduler

    def makemoduler(self, value):
        if value < self.moduler:
            return self.moduler
        elif value == self.moduler:
            return 0
        else:
            return value % self.moduler

    def moduleraddition(self, a, b):
        sum_mod = (a % self.moduler + b % self.moduler) % self.moduler
        if sum_mod == self.moduler:
            return 0
        return sum_mod

    def modulermultiplication(self, a, b):
        product_mod = (a % self.moduler) * (b % self.moduler) % self.moduler
        if product_mod == self.moduler:
            return 0
        return product_mod
