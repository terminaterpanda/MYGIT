import dill
import time
import random
import numpy as np
from sklearn.metrics import roc_curve, auc

import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn


from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


RANDOM_SEED = 2020
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class LSTMClassifier(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, hidden_size, num_layers, pad_idx
    ):
        super().__init__()
        self.embed_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        ) #생성할 embedding layer 크기 설정
        self.lstm_layer = nn.LSTM(
            
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=0.5           
        )
        self.last_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        embed_x = self.embed_layer(x)
        #숫자로 이루어진 token을 input으로 받는다고 가정
        output, (_, _) = self.lstm_layer(embed_x)

        last_output = output[:, -1, :]
        #(배치 크기, 문장길이, output_size) 여기서 가장 마지막 단어의 결과값을 사용
        last_output = self.last_layer(last_output)
        #문장의 가장 마지막 단어의 output을 넣어서 확률값을 calculate
        return last_output