import os
import re
import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mecab import MeCab
from gensim.models import CoherenceModel
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.simplefilter("ignore")
class Textmining:
    def __init__(self, file_paths, file_paths1, file_paths2):
        self.mecab = MeCab()
        self.file_paths = file_paths
        self.file_paths1 = file_paths1
        self.file_paths2 = file_paths2
        self.stopwords = []
        self.data = None
        self.data1 = None
        self.data2 = None
        
    def load_data(self):
        self.data = self._load_files(self.file_paths)
        
    def load_data2(self):
        self.data1 = self._load_files(self.file_paths1)
        
    def load_data3(self):
        self.data2 = self._load_files(self.file_paths2)
        
    def _load_files(self, file_paths):
        all_data = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                file = pd.read_excel(file_path, engine="openpyxl")
                all_data.append(file)
            else:
                raise ValueError("error 000")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            raise ValueError("error 001")
    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        return re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)).strip()
    
    def tokenize_text(self, data, column_name):
        if column_name in data.columns:
            data[f"{column_name}_tokens"] = data[column_name].apply(
                lambda x: [word for word in self.mecab.morphs(self.clean_text(x)) if 
                           word not in self.stopwords]
            )
            corpus = sum(data[f"{column_name}_tokens"].tolist(), [])
            return corpus
        else:
            raise ValueError("error 002")
        
    def compute_perplexity_graph(self, corpus, min_topics=2, max_topics=20):
        self.dictionary = gensim.corpora.Dictionary([corpus])
        bow_corpus = [self.dictionary.doc2bow(text) for text in [corpus]]
        perplexities = []
        topics_range = range(min_topics, max_topics+1)
        
        for num_topics in topics_range:
            lda_model = gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=self.dictionary, passes=10)
            perplexity = lda_model.log_perplexity(bow_corpus)
            perplexities.append(perplexity)
            
        plt.figure(figsize=(8, 5))
        plt.plot(topics_range, perplexities, marker='o')
        plt.xlabel("Number of Topics")
        plt.ylabel("Perplexity")
        plt.title("Perplexity Graph for Optimal Number of Topics")
        plt.show()
    def coherence_perplexity(self, corpus, min_topic=2, max_topic=20):
        coherence_values = []
        for i in range(min_topic, max_topic):
            bow_corpus = [self.dictionary.doc2bow(text) for text in [corpus]]
            ldamodel1 = gensim.models.LdaModel(bow_corpus, num_topics=i, id2word=self.dictionary, passes=10)
            coherence_model_lda = CoherenceModel(model=ldamodel1, texts=corpus, dictionary=self.dictionary, topn=10)
            coherence_lda = coherence_model_lda.get_coherence()
            coherence_values.append(coherence_lda)
        x = range(min_topic, max_topic)
        plt.plot(x, coherence_values)
        plt.xlabel("number of topics")
        plt.ylabel("coherence_score")
        plt.show()            
    def generate_wordcloud_bar(self, data, column_name, save_path):
        if column_name in data.columns:
            text = " ".join(data[column_name].dropna().astype(str)).replace(",", " ")
            words = text.split()
            word_counts = Counter(words)
            wordcloud = WordCloud(font_path="/Users/iseong-yong/Library/Fonts/GmarketSansMedium.otf", background_color='white',
                                  width=800, height=400).generate(text)
            plt.figure(figsize=(10, 5))
            #그림의 크기 설정(10, 5)
            plt.imshow(wordcloud, interpolation='bilinear')
            #그림을 화면에 출력 + interpolation = 부드러운 이미지 표현)
            plt.axis("off")
            #x, y축 숨김
            plt.savefig(save_path + "_wordcloud.png")
            plt.show()    
            top_words = pd.DataFrame(word_counts.most_common(20), columns=["단어", "빈도"])
            #단어와 빈도 리스트로 반환, 데이터프레임으로 변환
            sns.barplot(data=top_words, x = "단어", y = "빈도")
            plt.xticks(rotation=45)
            #x축의 단어라벨을 45도 회전
            plt.savefig(save_path + "_barchart.png")
            plt.show()           
        else:
            raise ValueError("error 003")
        
    def compute_tfidf_top_n(self, data, column_name, top_n=20):
        if f"{column_name}_tokens" in data.columns:
            corpus = [" ".join(tokens) for tokens in data[f"{column_name}_tokens"] if tokens]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0]))
            sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return pd.DataFrame(sorted_tfidf, columns=["단어", "TF-IDF 점수"])
        else:
            print("error 004")
        return None
    
    def run_analysis(self, save_dir="./graph_texts"):
        os.makedirs(save_dir, exist_ok=True)
        self.load_data()
        self.load_data2()
        self.load_data3()      
        
        for dataset, name in zip([self.data, self.data1, self.data2], ["중도", "진보", "보수"]):
            if dataset is not None:
                if "본문" in dataset.columns:
                    corpus = self.tokenize_text(dataset, "본문")
                    self.compute_perplexity_graph(corpus)
                    self.coherence_perplexity(corpus)
                    self.generate_wordcloud_bar(dataset, "본문", os.path.join(save_dir, f"wordcloud_{name}"))
                    lda_model = LatentDirichletAllocation(n_components=5, random_state=43)
                    lda_model.fit(bow_matrix)                    
                    vectorizer1 = CountVectorizer()
                    corpus_sentences = [" ".join(tokens) for tokens in dataset[f"본문_tokens"] if tokens]
                    bow_matrix = vectorizer1.fit_transform(corpus_sentences)
                if "키워드" in dataset.columns:
                    self.generate_wordcloud_bar(dataset, "키워드", os.path.join(save_dir, f"keywords_{name}"))
                
                if "특성추출(가중치순 상위 50개)" in dataset.columns:
                    self.generate_wordcloud_bar(dataset, "특성추출(가중치순 상위 50개)", os.path.join(save_dir, f"features_{name}"))
        tfidf_results = {
            "중도": self.compute_tfidf_top_n(self.data, "본문") if self.data is not None else None,
            "진보": self.compute_tfidf_top_n(self.data1, "본문") if self.data1 is not None else None,
            "보수": self.compute_tfidf_top_n(self.data2, "본문") if self.data2 is not None else None,
        }
        for key, df in tfidf_results.items():
            #tfidf_results = {중도:df1, 진보:df2, 보수:df3}
            #items()를 사용해서 kdt, df로 중도, 진보, 보수와 해당하는 dataframe을 가지고 옴
            if df is not None:
                df.to_csv(os.path.join(save_dir, f"tfidf_{key}.csv"), index=False, encoding="utf-8-sig"
                          )
        print("finished")
    
file_paths = ["/Users/iseong-yong/Desktop/files/news/news_경향.xlsx"] #파일 경로
file_paths1 = ["/Users/iseong-yong/Desktop/files/news/news_문화.xlsx"]#파일 경로
file_paths2 = ["/Users/iseong-yong/Desktop/files/news/news_중앙.xlsx"]#파일 경로
tm = Textmining(file_paths, file_paths1, file_paths2)
tm.run_analysis()