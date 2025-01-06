import os
import numpy as np
from collections import defaultdict
from konlpy.tag import Mecab
#defaultdict == 값이 없는 키를 호출해도 기본 값 반환 딕셔너리 클래스

MecabTokenizer = Mecab()
def make_save_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
#경로 존재하지 않을 시 디렉토리 생성 함수
def get_tokenizer(tokenizer_name):
    if tokenizer_name == "mecab":
        # Replace with an actual Mecab tokenizer instantiation
        return MecabTokenizer()  # Placeholder
    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")
#토크나이저 반환함수
class CbowModel:
    def __init__(self,  train_frame, embedding_frame, model_frame,
                 embedding_corpus_frame, embedding_method="fasttext",
                 is_weighted=True, average=False, dim=100, tokenizer_name="mecab"):
        # Configurations
        make_save_path(model_frame) #모델 디렉토리가 없는 경우에 생성하는 코드
        self.dim = dim
        self.average = average
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.embeddings = {} #딕셔너리 초기화

        model_full_name = model_frame + ("_weighted" if is_weighted else "_original")
        #가중치 적용 여부에 따라서 모델 이름을 설정
        if is_weighted:
            self.embeddings = self.load_or_construct_weighted_embedding(
                embedding_frame, embedding_method, embedding_corpus_frame
            )
            print("Weighted embeddings loaded.")
        else:
            self.embeddings = self.load_word_embeddings(embedding_frame, embedding_method)
            print("Original embeddings loaded.")
        #가중치 적용 임베딩 로드 혹은 생성
        if not os.path.exists(model_full_name):
            print("Training model.")
            self.model = self.train_model(train_frame, model_full_name)
        else:
            print("Loading model.")
            self.model = self.load_model(model_full_name)
        #없는 경우에 학습
    def compute_word_frequency(self, embedding_corpus_frame):
        total_count = 0
        words_count = defaultdict(int)
        with open(embedding_corpus_frame, "r") as f:
            for line in f:
                tokens = line.strip().split()
                for token in tokens:
                    words_count[token] += 1
                    total_count += 1
        return words_count, total_count
        #코퍼스에서 단어의 빈도를 계산 + 반환(각 단어의 등장수와 전체 단어수 반환)
    def load_or_construct_weighted_embedding(self, embedding_frame, embedding_method, embedding_corpus_frame, a=0.0001):
        dictionary = {}
        weighted_file = embedding_frame + "-weighted"

        if os.path.exists(weighted_file):
            with open(weighted_file, "r") as f:
                for line in f:
                    word, weighted_vector = line.strip().split("\u241E")
                    dictionary[word] = np.array([float(el) for el in weighted_vector.split()])
        else:
            words, vecs = self.load_word_embeddings(embedding_frame, embedding_method)
            word_count, total_word_count = self.compute_word_frequency(embedding_corpus_frame)

            with open(weighted_file, "w") as f:
                for word, vec in zip(words, vecs):
                    word_prob = word_count.get(word, 0) / total_word_count
                    weighted_vector = (a / (word_prob + a)) * np.asarray(vec)
                    dictionary[word] = weighted_vector
                    f.write(word + "\u241E" + " ".join(map(str, weighted_vector)) + "\n")

        return dictionary
        #임베딩 파일이 존재하면 로드하고 없으면 가중치를 적용하여 생성 후 저장하기
    def load_word_embeddings(self, embedding_frame, embedding_method):
        words, vectors = [], []
        with open(embedding_frame, "r") as f:
            for line in f:
                parts = line.strip().split()
                words.append(parts[0])
                vectors.append([float(x) for x in parts[1:]])
        return words, np.array(vectors)
        #주어진 임베딩 파일에서 단어와 벡터를 읽어오는 함수
    def train_model(self, train_data_frame, model_frame):
        model = {"vectors": [], "labels": [], "sentences": []}
        train_data = self.load_or_tokenize_corpus(train_data_frame)

        with open(model_frame, "w") as f:
            for sentence, tokens, label in train_data:
                tokens = self.tokenizer.morphs(sentence)
                sentence_vector = self.get_sentence_vector(tokens)

                model["sentences"].append(sentence)
                model["vectors"].append(sentence_vector)
                model["labels"].append(label)

                str_vector = " ".join(map(str, sentence_vector))
                f.write(f"{sentence}\u241E{' '.join(tokens)}\u241E{str_vector}\u241E{label}\n")

        return model
        #학습 data 토큰화 + 각 문장에 대한 벡터 생성 모델 학습
    def get_sentence_vector(self, tokens):
        vector = np.zeros(self.dim)
        for token in tokens:
            if token in self.embeddings:
                vector += self.embeddings[token]
        if self.average and len(tokens) > 0:
            vector /= len(tokens)
        vector_norm = np.linalg.norm(vector)
        return vector / vector_norm if vector_norm != 0 else vector
        #토큰 리스트를 받아서 문장 벡터를 생성
    def predict(self, sentence):
        tokens = self.tokenizer.morphs(sentence)
        sentence_vector = self.get_sentence_vector(tokens)
        scores = np.dot(self.model["vectors"], sentence_vector)
        pred_index = np.argmax(scores)
        return self.model["labels"][pred_index]
        #문장 벡터와 모델 벡터간의 유사도 계산(코사인 유사도)
    def predict_by_batch(self, tokenized_sentences, labels):
        sentence_vectors = [self.get_sentence_vector(tokens) for tokens in tokenized_sentences]
        scores = np.dot(self.model["vectors"], np.array(sentence_vectors).T)
        preds = np.argmax(scores, axis=0)

        eval_score = sum(
            1 for pred, label in zip(preds, labels) if self.model["labels"][pred] == label
        )
        return preds, eval_score
        #점수에서 가장 큰 값을 가지는 레이블 반환

#######################ELMO embedding
Vocabulary = int(12)
class UnicodeCharsVocabulary(Vocabulary):
    #문자 수준에서 작동, 단어를 개별 문자로 쪼개고 각 문자를 고유한 ID로 변환
    def __init__(self, filename, max_word_length, **kwargs):
        #filename == 단어 사전을 로드하는 파일 이름
        #max_word_length == 단어를 변환할 당시 생성되는 문자 배열 최대 길이 설정
        #kwargs == 부모 클래스의 추가 옵션을 전달
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length
        self.bos_char = 256 # <begin sentence>
        self.eos_char = 257 # <end sentence>
        self.bow_char = 258 # <begin word>
        self.eow_char = 259 # <end word>
        self.pad_char = 260 # <padding>  패딩 문자로 사용되는 id
        
    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self._max_word_length], dtype=np.int32)
        code[:] = self.pad_char
        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char
        return code
    
