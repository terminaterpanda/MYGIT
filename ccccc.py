from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
from networkx.algorithms import community
import warnings
import numpy as np
import pandas as pd
import os
warnings.simplefilter('ignore')
font_path = "/Users/iseong-yong/Library/Fonts/MALGUN.TTF"
if not os.path.exists(font_path):
    raise ValueError("error 001")
font_prop = fm.FontProperties(fname=font_path)
plt.rc("font", family = font_prop.get_name())
class Textmining:
    def __init__(self, file_path):
        self.file_paths = file_path
        self.stopwords = ["대통령", "계엄"]
        self.data = None
        self.dtm = None
#연결중심성을 사용해서 의미연결망 분석 -> 색깔 pallete apply
#상위단어출현(50개) use. #50개로 하고 노드 크기 -> 비설정
#layout apply. -> 조굼 더 예쁘게?
#전체 말뭉치를 다 합쳐서 꺾은선그래프 생성 -> 하나 커다란 것도 따로 만들어야함
#코드들을 다 구현해놓고, file_path를 list로 해서 자동 처리할 수 있게 코드를 하나 더 구현해놓아야함.
    def load_data(self):
        self.data = self.load_files(self.file_paths)
        if self.data is None:
            raise ValueError("❌ 데이터를 불러오지 못했습니다. 파일 경로를 확인하세요.")

    def load_files(self, file_paths):
        all_data = []
        for path in file_paths:
            if os.path.exists:
                file = pd.read_excel(path, engine="openpyxl")
                all_data.append(file)
            else:
                raise ValueError("error 003")
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            raise ValueError("error 004")
        
                
    def make_coherence_score(self):
        if "키워드" in self.data.columns:  # 조건문 수정
            self.data["키워드"] = self.data["키워드"].fillna("").astype(str)
            self.data["키워드"] = self.data["키워드"].apply(
                lambda x: ",".join(
                    [word.strip() for word in x.split(",") if word.strip() and word.strip() not in self.stopwords]
                )
            )
        else:
            raise KeyError("❌ '키워드' 컬럼이 데이터에 없습니다.")

    def create_dtm(self, max_features=1000):
        vectorizer = CountVectorizer(
            tokenizer=lambda x: x.split(","), 
            binary=True,
            max_features=max_features  # 최대 단어 개수 제한
        )
        self.dtm = vectorizer.fit_transform(self.data["키워드"])
        dtm_df = pd.DataFrame(self.dtm.toarray(), columns=vectorizer.get_feature_names_out())
        print(dtm_df.loc[1])
        return dtm_df
    def create_coocurence_score(self, top_n = 50):
        if self.dtm is None:
            raise ValueError("error 001")
        co_matrix = (self.dtm.T @ self.dtm).toarray()
        np.fill_diagonal(co_matrix, 0)
        terms = list(self.create_dtm().columns)
        word_freq = self.dtm.toarray().sum(axis = 0)
        print("starting")
        top_indices = np.argsort(word_freq)[::-1][:top_n]
        top_terms = [terms[i] for i in top_indices]
        filtered_matrix = co_matrix[top_indices][:, top_indices]
        cooc_df = pd.DataFrame(filtered_matrix, index=top_terms, columns=top_terms)
        
        G = nx.Graph()
        for i, term1 in enumerate(top_terms):
            for j, term2 in enumerate(top_terms):
                if i < j and cooc_df.iloc[i, j] > 0:
                    G.add_edge(term1, term2, weight=cooc_df.iloc[i, j])
        
        closeness_centrality = nx.closeness_centrality(G)
        communities = list(community.greedy_modularity_communities(G))
        community_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_dict[node] = i
        plt.figure(figsize=(60, 80))
        pos = nx.spring_layout(G, k = 8,iterations=200,seed=42)
        node_size = [closeness_centrality[node] * 3500 for node in G.nodes()]
        node_colors = [community_dict[node] for node in G.nodes()]
        cmap = plt.cm.get_cmap('tab20', len(set(node_colors)))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, cmap=cmap,node_color=node_colors, edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=12, font_family=font_prop.get_name())
        nx.draw_networkx_edges(G, pos, width= 1,alpha=1)
        edge_labels = {(u,v): str(d["weight"]) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red", font_family=font_prop.get_name())
        
        plt.title("동시출현 네트워크 그래프", fontsize=20)
        plt.show()
        return G    
    
tm = Textmining(["/Users/iseong-yong/Desktop/경제_계엄_1주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_2주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_3주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_4주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_5주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_6주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_7주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_8주차.xlsx",
                 "/Users/iseong-yong/Desktop/경제_계엄_9주차.xlsx"])
tm.load_data()
tm.make_coherence_score()
tm.create_dtm()
tm.create_coocurence_score()