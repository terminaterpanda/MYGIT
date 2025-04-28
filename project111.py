# 필요한 라이브러리 임포트
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

class Texter:
    def __init__(self, file_paths):
        self.file_paths = file_paths  # 4개 파일 경로 저장
        self.results = {}  # 각 파일별 결과 저장

    def textstart(self, file_path):
        """엑셀에서 데이터를 불러와 토큰화"""
        df = pd.read_excel(file_path, engine='openpyxl')
        documents = df['키워드'].astype(str).tolist()

        processed_docs = []
        for doc in documents:
            words = [word.strip() for word in doc.strip().split(',') if word.strip()]
            processed_docs.append(words)

        print(f"파일 {file_path} 처리 완료. 예시 문서(토큰화 결과):", processed_docs[0])
        return processed_docs

    def lda_topic(self, processed_docs):
        """토픽 모델링을 위한 사전 및 코퍼스 생성"""
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(text) for text in processed_docs]
        return dictionary, corpus

    def lda_training(self, dictionary, corpus, processed_docs):
        """LDA 모델 학습 및 최적의 토픽 수 찾기"""
        coherence_scores = []
        perplexity_scores = []
        models = {}

        print("\n토픽 수 별 평가 결과:")
        for num_topics in range(5, 11):
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10
            )
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=processed_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            perplexity = lda_model.log_perplexity(corpus)

            coherence_scores.append(coherence_score)
            perplexity_scores.append(perplexity)
            models[num_topics] = lda_model

            print(f"토픽 수: {num_topics:2d} | Coherence Score: {coherence_score:.4f} | Log Perplexity: {perplexity:.4f}")

        # 최적 토픽 수 선택
        best_idx = coherence_scores.index(max(coherence_scores))
        optimal_num_topics = list(range(5, 11))[best_idx]
        optimal_model = models[optimal_num_topics]

        print("\n최적의 토픽 수 (coherence score 기준):", optimal_num_topics)

        # 토픽별 상위 단어 추출
        topic_words = {}
        for idx, topic in optimal_model.show_topics(num_topics=optimal_num_topics, num_words=10, formatted=False):
            topic_words[f"Topic {idx+1}"] = [word for word, _ in topic]

        return pd.DataFrame(topic_words)

    def run(self):
        """전체 프로세스 실행"""
        for file_path in self.file_paths:
            print(f"\n=== {file_path} 처리 시작 ===")
            processed_docs = self.textstart(file_path)
            dictionary, corpus = self.lda_topic(processed_docs)
            df_topics = self.lda_training(dictionary, corpus, processed_docs)

            self.results[file_path] = df_topics

            # 결과를 Google Drive에 저장
            output_path = f"/content/drive/MyDrive/시험)분석용 데이터/분석결과/3. 윤석열/9주차{os.path.basename(file_path).split('.')[0]}.xlsx"
            df_topics.to_excel(output_path, index=False)
            print(f"파일 저장 완료: {output_path}")

        print("\n모든 파일 처리 완료!")

# 사용 예시
file_paths = [
    "/content/drive/MyDrive/시험)분석용 데이터/3. 윤석열/9주차_01.27~02.02/전국_윤석열_9주차.xlsx",
    "/content/drive/MyDrive/시험)분석용 데이터/3. 윤석열/9주차_01.27~02.02/경제_윤석열_9주차.xlsx",
    "/content/drive/MyDrive/시험)분석용 데이터/3. 윤석열/9주차_01.27~02.02/방송사_윤석열_9주차.xlsx",
    "/content/drive/MyDrive/시험)분석용 데이터/3. 윤석열/9주차_01.27~02.02/인터넷신문_윤석열_9주차.xlsx"
]

texter = Texter(file_paths)
texter.run()
