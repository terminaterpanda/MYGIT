import os
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from matplotlib import rcParams
warnings.simplefilter("ignore")

# 폰트 설정 (한글 폰트 적용)
fe = fm.FontEntry(fname="/content/drive2/MyDrive/GmarketSansTTFBold.ttf", name="GmarketSansTTFBold")
fm.fontManager.ttflist.insert(0, fe)
rcParams["font.family"] = "GmarketSansTTFBold"
matplotlib.rcParams["axes.unicode_minus"] = False

class Textmining:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stopwords = ["대통령", "윤석열", "계엄"]
        self.data = None

    def load_data(self):
        self.data = self._load_files(self.file_path)

    def _load_files(self, file_paths):
        all_data = []
        for path in file_paths:
            if os.path.exists(path):
                file = pd.read_excel(path, engine="openpyxl")
                all_data.append(file)
            else:
                raise ValueError("error: 001: no data compiled")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            raise ValueError("error: 002: no data loaded")

    def compute_tfidf_top_n(self, data, column_name, top_n=20):
        if column_name in data.columns:
            corpus = data[column_name].dropna().astype(str).tolist()
            vectorizer = TfidfVectorizer(stop_words=self.stopwords, use_idf=True, smooth_idf=True)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0]))
            sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return pd.DataFrame(sorted_tfidf, columns=["단어", "TF-IDF 점수"])
        else:
            print("error : no col in data")
        return None

    def generate_wordcloud_bar(self, data, column_name, save_dir):
        if column_name in data.columns:
            os.makedirs(save_dir, exist_ok=True)

            all_words = []
            for row in data[column_name].dropna().astype(str).tolist():
                tokens = [token.strip() for token in row.split(",") if token.strip() and token.strip() not in self.stopwords]
                all_words.extend(tokens)

            text = " ".join(all_words)
            word_counts = Counter(all_words)
            wc = WordCloud(
                font_path="/content/drive2/MyDrive/GmarketSansTTFBold.ttf",
                background_color='white',
                width=800,
                height=400,
                max_words=100
            ).generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout()
            wordcloud_path = os.path.join(save_dir, "wordcloud.png")
            plt.savefig(wordcloud_path)
            plt.show()
            #bar chart generate
            top_words = pd.DataFrame(word_counts.most_common(20), columns=["단어", "빈도"])
            plt.figure(figsize=(10, 5))
            palette = sns.color_palette("coolwarm", n_colors=len(top_words))
            ax = sns.barplot(data=top_words, x="단어", y="빈도", palette=palette)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height + 1),
                    ha='center', va='bottom', fontsize=10, color="black", fontweight='bold'
                )
            plt.xticks(rotation=45, ha='right')
            plt.xlabel("단어")
            plt.ylabel("빈도")
            plt.tight_layout()
            bar_chart_path = os.path.join(save_dir, "barchart.png")
            plt.savefig(bar_chart_path)
            plt.show()

        else:
            raise ValueError("error : the selected col is not in dataset.")

    def run_analysis(self, save_dir="/content/drive2/MyDrive/tf-idf_경제_윤석열_9주차"):
        os.makedirs(save_dir, exist_ok=True)
        self.load_data()
        tfidf_result = self.compute_tfidf_top_n(self.data, "키워드") if self.data is not None else None
        self.generate_wordcloud_bar(self.data, "키워드", save_dir) if self.data is not None else None
        if tfidf_result is not None:
            tfidf_result.to_csv(os.path.join(save_dir, "tfidf_result.csv"), index=False, encoding="utf-8-sig")

    def run_analysisa(self, save_dir="/content/drive2/MyDrive/tf-idf_방송사_윤석열_9주차"):
        os.makedirs(save_dir, exist_ok=True)
        self.load_data()
        tfidf_result = self.compute_tfidf_top_n(self.data, "키워드") if self.data is not None else None
        self.generate_wordcloud_bar(self.data, "키워드", save_dir) if self.data is not None else None
        if tfidf_result is not None:
            tfidf_result.to_csv(os.path.join(save_dir, "tfidf_result.csv"), index=False, encoding="utf-8-sig")

    def run_analysisb(self, save_dir="/content/drive2/MyDrive/tf-idf_인터넷신문_윤석열_9주차"):
        os.makedirs(save_dir, exist_ok=True)
        self.load_data()
        tfidf_result = self.compute_tfidf_top_n(self.data, "키워드") if self.data is not None else None
        self.generate_wordcloud_bar(self.data, "키워드", save_dir) if self.data is not None else None
        if tfidf_result is not None:
            tfidf_result.to_csv(os.path.join(save_dir, "tfidf_result.csv"), index=False, encoding="utf-8-sig")

    def run_analysisc(self, save_dir="/content/drive2/MyDrive/tf-idf_전국_윤석열_9주차"):
        os.makedirs(save_dir, exist_ok=True)
        self.load_data()
        tfidf_result = self.compute_tfidf_top_n(self.data, "키워드") if self.data is not None else None
        self.generate_wordcloud_bar(self.data, "키워드", save_dir) if self.data is not None else None
        if tfidf_result is not None:
            tfidf_result.to_csv(os.path.join(save_dir, "tfidf_result.csv"), index=False, encoding="utf-8-sig")
file_paths = ["/content/drive2/MyDrive/경제_윤석열_9주차.xlsx"]
file_pathsa = ["/content/drive2/MyDrive/방송사_윤석열_9주차.xlsx"]
file_pathsb = ["/content/drive2/MyDrive/인터넷신문_윤석열_9주차.xlsx"]
file_pathsc = ["/content/drive2/MyDrive/전국_윤석열_9주차.xlsx"]
tm = Textmining(file_paths)
tma = Textmining(file_pathsa)
tmb = Textmining(file_pathsb)
tmc = Textmining(file_pathsc)

tm.run_analysis()
tma.run_analysisa()
tmb.run_analysisb()
tmc.run_analysisc()
