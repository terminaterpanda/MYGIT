import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from mecab import MeCab

class TextMining:
    def __init__(self, file_paths):
        self.mecab = MeCab()         #형태소 분석
        self.file_paths = file_paths #file_paths로 file_path 복수형으로 받음
        self.data = self.load_data() #self.data == self.load_data()
        self.stopwords = []          #stopwords를 지정
        self.processed_data = {}     #self.processed_data
    
    def load_data(self):
        all_data = [] #데이터 리스트
        for file_path in self.file_paths:
            if os.path.exists(file_path): #os.path가 존재하는 경우
                df = pd.read_excel(file_path, engine="openpyxl")
                df['source'] = os.path.basename(file_path)
                all_data.append(df)
            else:
                print(f"❌ 파일을 찾을 수 없음: {file_path}")
        return pd.concat(all_data, ignore_index=True) if all_data else None
    #self.인자를 사용하지 않으므로 @staticmethod use
    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        return re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)).strip()
    
    def tokenize_text(self, column_name):
        if column_name not in  ["본문", "키워드", "특성추출(가중치순 상위 50개"]:
            raise ValueError("error 002")
        if column_name in self.data.columns:
            self.data[column_name] = self.data[column_name].fillna("")
            #결측값을 빈 문자열로 채우기
            self.data[f"{column_name}_tokens"] = self.data[column_name].apply(
                lambda x: [word for word in self.mecab.morphs(self.clean_text(x)) if word not in self.stopwords])
            
            corpus = sum(self.data[f"{column_name}_tokens"].tolist(), [])
            return self.data[[column_name, f"{column_name}_tokens"]], corpus
        else:
            raise ValueError(f"Error: {column_name} 컬럼이 없습니다.")
    
    def generate_wordcloud(self, column_name, save_path):
        #column_name == 워드클라우드 make col
        #주어진 컬럼의 text data를 워드클라우드로 시각화 및 파일 저장
        if column_name in self.data.columns:
            text = " ".join(self.data[column_name].dropna().astype(str))
            wordcloud = WordCloud(font_path="/Users/iseong-yong/Library/Fonts/GmarketSansMedium.otf", background_color='white', width=800, height=400).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig(save_path)
            print(f"✅ 워드클라우드 저장 완료: {save_path}")
            plt.show()
        else:
            print(f"❌ {column_name} 컬럼이 존재하지 않습니다.")
    
    def compute_tfidf_top_n(self, column_name, top_n=20):
        if f"{column_name}_tokens" in self.data.columns:
            corpus = [" ".join(tokens) for tokens in self.data[f"{column_name}_tokens"] if tokens]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0]))
            sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return pd.DataFrame(sorted_tfidf, columns=["단어", "TF-IDF 점수"])
        else:
            print(f"❌ {column_name}_tokens 컬럼이 존재하지 않습니다.")
        return None
    
    def analyze_keyword_frequency(self):
        if "키워드" in self.data.columns:
            keywords = " ".join(self.data["키워드"].dropna().astype(str)).split()
            keyword_counts = Counter(keywords)
            top_keywords = pd.DataFrame(keyword_counts.most_common(10), columns=["단어", "빈도"])
            print("✅ 키워드 상위 10개 분석 완료")
            return top_keywords
        else:
            print("error 003")
            return None
    
    def run_analysis(self, save_dir="./results"):
        os.makedirs(save_dir, exist_ok=True)
        
        if "본문" in self.data.columns:
            self.tokenize_text("본문")
            self.generate_wordcloud("본문", os.path.join(save_dir, "wordcloud_main.png"))
            tfidf_df = self.compute_tfidf()
            tfidf_df.to_csv(os.path.join(save_dir, "tfidf_results.csv"), index=False, encoding="utf-8-sig")
            print(f"✅ TF-IDF 데이터 저장 완료: {os.path.join(save_dir, 'tfidf_results.csv')}")

        for column in self.data.columns:
            if column not in ["본문", "source"] and self.data[column].dtype == object:
                self.generate_wordcloud(column, os.path.join(save_dir, f"wordcloud_{column}.png"))
        
        top_keywords_df = self.analyze_keyword_frequency()
        if top_keywords_df is not None:
            top_keywords_df.to_csv(os.path.join(save_dir, "top_keywords.csv"), index=False, encoding="utf-8-sig")
            print(f"✅ 키워드 빈도 데이터 저장 완료: {os.path.join(save_dir, 'top_keywords.csv')}")
        
        print("🚀 분석 완료!")

# 사용 예시
file_paths = ["/Users/iseong-yong/Downloads/NewsResult_20250206-20250207.xlsx"]
text_miner = TextMining(file_paths)
text_miner.run_analysis()

import os
import re
import pandas as pd
from mecab import MeCab

class TextMining:
    def __init__(self, file_paths):
        self.mecab = MeCab()
        self.file_paths = file_paths
        self.stopwords = []  # 불용어 리스트
        self.data = None  # 데이터프레임 초기화

    def load_data(self):
        all_data = []
        for file_path in self.file_paths:
            if os.path.exists(file_path):
                file = pd.read_excel(file_path, engine="openpyxl")
                all_data.append(file)
            else:
                raise ValueError("error001: 파일을 찾을 수 없습니다.")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)  # 여러 파일을 합치기
        else:
            raise ValueError("error002: 데이터가 비어 있습니다.")

    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        return re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)).strip()
    
    def tokenize_column(self, target_column="본문"):
        if self.data is None:
            raise ValueError("error003: 데이터를 먼저 로드하세요.")
        
        if target_column not in self.data.columns:
            raise ValueError(f"error004: '{target_column}' 열이 데이터에 존재하지 않습니다.")
        
        # 텍스트 전처리 및 토큰화 수행
        self.data[f"{target_column}_tokens"] = self.data[target_column].apply(
            lambda x: [word for word in self.mecab.morphs(self.clean_text(x)) if word not in self.stopwords]
        )
        
        # 모든 토큰을 하나의 리스트(말뭉치)로 결합
        corpus = sum(self.data[f"{target_column}_tokens"].tolist(), [])
        
        return self.data[[target_column, f"{target_column}_tokens"]], corpus