import numpy as np
import re
from konlpy.tag import Okt

class tokenizer():
    def __init__(self, texter, n_gram):
        self.texter = texter
        self.okt = Okt
        self.n_gram = n_gram
        self.result = None
        self.token_list = None
        
    def texdata_preprocessing(self):
        korean = re.sub(r"[^가-힣\s]", "", self.texter)
        korean = korean.strip()
        self.texter = korean
        return self.texter
    #self.texter에서 볼 수 있듯이, self.texter 는 데이터 정제된 걸 use
    
    
    def tokenizer(self):
        if not self.texter:
            raise ValueError("error 01")
        tokens = self.okt.morphs(self.texter)
        self.token_list = tokens
        return self.token_list
        #self.token_list = preprocessed된 data를 토큰화
    
      
    def make_list(self, n_gram):      
        if not self.token_list:
            raise ValueError("error 001")
        lister = len(self.token_list)
        self.lister = lister
        self.n_gram = n_gram
        #n_gram변수를 self.에서 참조할 수 있게 만들어줌
        self.n_gram_list = [
            self.token_list[i:i + n_gram]
            for i in range(lister - n_gram + 1)
        ]
        return self.n_gram_list
    
    def one_hot_encoding(self, tokens):
        unique_tokens = list(set(tokens))
        #set(tokens) = set는 토큰에서 중복을 제거하고 list를 생성함 set == "집합"
        tokens_to_index = {token: idx for idx, token in enumerate(unique_tokens)}
        #idx for idx = 인덱스를 만들기 위한  것 고유 토큰들을 인덱싱하여 토큰 인덱스 딕셔너리 생성
        one_hot_matrix = np.eye(len(unique_tokens))[list(map(tokens_to_index.get, tokens))]
        #단위행렬과 원핫 인코디ㅏㅇ 행렬을 만듬
        return one_hot_matrix, tokens_to_index
    
    def calculate_vector(self):
        if not self.n_gram_list:
            raise ValueError("error 002")
        total_vector = np.zeros(len(self.token_list))
        #token_lists크기만큼 0으로 인코딩된 vector를 생성
        
        for n_gram in self.n_gram_list:
            one_hot_matrix, _ = self.one_hot_encoding(n_gram)
            #n_gram에 대해서 원핫 행렬 생성
            product_vector = np.prod(one_hot_matrix, axis=0) 
            # Element-wise product(원소 곱을 계산하는게 np.prod)
            
            total_vector += product_vector  # Sum up the vectors
            #이 벡터를 더해서 tanh에 삽입
            
            encoded_vector = np.tanh(total_vector)
            return encoded_vector
        
    def iterative_process(self, target_vector, max_iterations=1000, tolerance=1e-6):
        current_vector = self.calculate_vector()
        iteration = 0
        
        while iteration < max_iterations:
            diff = np.linalg.norm(current_vector - target_vector)
            if diff < tolerance:
                break
            current_vector = np.tanh(current_vector + target_vector)
            iteration += 1
            
        return current_vector, iteration
    
    #반복 process
        """
        이제 여기 self.token_list에서 되어 있는 단어들만큼의 크기로 list를 만들고, n_gram
        개수만큼 개수를 지정한 후 참조할 행렬을 지정.
        그 이후에 그 행렬에서 원핫 인코딩 벡터를 만들어서 각각 곱한 후, 그 행렬들을 더함.
        그 후 커다란 벡터를 하이퍼탄젠트에 넣은 후, 결과값 인코딩 벡터를 만들고 값의 차이를 계산해서
        무한반복하는 code를 만들어내야함.
        """

            
    
    