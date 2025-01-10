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
    
    def tokenizer(self):
        if not self.texter:
            raise ValueError("error 01")
        tokens = self.okt.morphs(self.texter)
        self.token_list = tokens
        return self.token_list
    
    
        """
        이제 여기 self.token_list에서 되어 있는 단어들만큼의 크기로 list를 만들고, n_gram
        개수만큼 개수를 지정한 후 참조할 행렬을 지정.
        그 이후에 그 행렬에서 원핫 인코딩 벡터를 만들어서 각각 곱한 후, 그 행렬들을 더함.
        그 후 커다란 벡터를 하이퍼탄젠트에 넣은 후, 결과값 인코딩 벡터를 만들고 값의 차이를 계산해서
        무한반복하는 code를 만들어내야함.
        """
    
    