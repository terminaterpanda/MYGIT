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
        self.dictionary = None
    
    def load_data(self):
        self.data = self._load_files(self.file_paths)
    
    def load_data2(self):
        self.data1 = self._load_files(self.file_paths1)
    
    def load_data3(self):
        self.data2 = self._load_files(self.file_paths2)
    
    def _load_files(self, file_paths):
        all_data = []
        for file_path in file_paths:
            file_path = file_path.strip()
            if os.path.exists(file_path):
                file = pd.read_excel(file_path, engine="openpyxl")
                all_data.append(file)
            else:
                raise ValueError(f"error 000 - File not found: {file_path}")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            raise ValueError("error 001 - No data loaded")
    
    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        return re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)).strip()
    
    def tokenize_text(self, data, column_name):
        if column_name in data.columns:
            data[f"{column_name}_tokens"] = data[column_name].apply(
                lambda x: [word for word in self.mecab.morphs(self.clean_text(x)) if word not in self.stopwords]
            )
            corpus = data[f"{column_name}_tokens"].tolist()
            return corpus
        else:
            raise ValueError("error 002 - Column not found")
    
    def compute_perplexity_graph(self, corpus, min_topics=2, max_topics=20):
        self.dictionary = gensim.corpora.Dictionary(corpus)
        bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]
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
        bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]
        for i in range(min_topic, max_topic):
            ldamodel = gensim.models.LdaModel(bow_corpus, num_topics=i, id2word=self.dictionary, passes=10)
            coherence_model_lda = CoherenceModel(model=ldamodel, texts=corpus, dictionary=self.dictionary, topn=10)
            coherence_lda = coherence_model_lda.get_coherence()
            coherence_values.append(coherence_lda)
        
        plt.plot(range(min_topic, max_topic), coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.show()
    
    def run_analysis(self, save_dir="./graph_texts"):
        os.makedirs(save_dir, exist_ok=True)
        self.load_data()
        self.load_data2()
        self.load_data3()      
        
        for dataset, name in zip([self.data, self.data1, self.data2], ["중도", "진보", "보수"]):
            if dataset is not None and "본문" in dataset.columns:
                corpus = self.tokenize_text(dataset, "본문")
                self.compute_perplexity_graph(corpus)
                self.coherence_perplexity(corpus)
        
        print("Finished analysis")
    
file_paths = ["/Users/iseong-yong/Desktop/files/news/news_경향.xlsx"]
file_paths1 = ["/Users/iseong-yong/Desktop/files/news/news_문화.xlsx"]
file_paths2 = ["/Users/iseong-yong/Desktop/files/news/news_중앙.xlsx"]

tm = Textmining(file_paths, file_paths1, file_paths2)
tm.run_analysis()
