import pandas as pd
import numpy as np
import requests
#data scrap-ing 할때(requests)응답
import re
#정규식 import 
import time
#시간 import - data name 따로 지정하기 위해서 use.
from bs4 import BeautifulSoup
#data scraping 하는 package
from datetime import datetime

def clean_text(text):
    return re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)).strip()
#입력된 text에서 한글과 영어, 공백을 제외한 모든 문자를 제거
class Scraping:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        #scraping에 stack 형태로 넘겨줌
    def scrap_save(self):
        try:
            res = requests.get(self.url, headers=self.headers)
            #requests.get()을 사용해서 webpage를 요청함
            
            res.raise_for_status()  # status를 정의하여 가지고 올 수 있는지를 확인.

            soup = BeautifulSoup(res.text, "lxml")
            #html을 파싱해오기
            text = soup.get_text(separator="\n", strip=True)
            #+ strip=true로 순수한 text만 추출
            # \n을 사용해서 줄바꿈으로 seperator use.

            sentences = [clean_text(sentence) for sentence in text.split(".") if sentence.strip()]            
            #data 정제 공백 삭제
            #마침표를 기준으로 문장을 나누고 공백을 delete
            korean_sentences = [
                sentence for sentence in sentences if re.search(r"[가-힣]", sentence)
            ]
            #한글이 포함된 문장만 필터링
            df = pd.DataFrame(korean_sentences, columns=["sentence"])
            #한글 문장만 포함된 dataframe 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #timestamp -> 시간을 측정해서 file 이름을 넣고, 그런 상태로 filename.csv 로 저장.
            filename = f"korean_sentences_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            #파일이름을 filename으로 생성한후 csv파일로 저장(utf-8) 인코딩(제일 자주 사용)
            print(f"saved {filename}")
            self.df = df
            return df
        #오류 예외 처리
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            return pd.DataFrame()
        except Exception as err:
            print(f"An error occurred: {err}")
            return pd.DataFrame()

    def continuous_scrap(self, refresh_interval, min_data_count = 200):        
        collected_data = pd.DataFrame()#수집된 data저장할 dataframe
        while True:
            print("refresh")
            new_data = self.scrap_save()
            collected_data = pd.concat([collected_data, new_data]).drop_duplicates().reset_index(drop = True )

            if len(collected_data) >= min_data_count:
                print("데이터 수집완료")
                return collected_data
            print(f"{refresh_interval} 동안 대기")
            time.sleep(refresh_interval)
            #refresh_interval초 동안 대기 후 다시 스크래핑 반복
            
if __name__ == "__main__":
    url = "https://news.naver.com/section/100"
    headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}
    scraper=Scraping(url, headers)
    while True:
        scraper.continuous_scrap(refresh_interval=60, min_data_count=200)