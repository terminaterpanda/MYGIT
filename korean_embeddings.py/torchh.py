import torch

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0]
])
w_query = torch.tensor([
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
])
w_key = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])
w_value = torch.tensor([
    [0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0],
    [1.0, 0.0, 3.0],
    [1.0, 1.0, 0.0]
])

keys = torch.matmul(x, w_key)
querys = torch.matmul(x, w_query)
values = torch.matmul(x, w_value)
#torch.matmul = "행렬곱을 수행하는 함수"

attn_scores = torch.matmul(querys, keys.T)

import numpy as np
from torch.nn.functional import softmax
key_dim_sqrt = np.sqrt(keys.shape[-1])
attn_probs = softmax(attn_scores / key_dim_sqrt, dim = -1)

weighted_values = torch.matmul(attn_probs, values)

import torch
x = torch.tensor([2,1])
w1 = torch.tensor([[3, 2, -4],[2,-3,1]])
b1 = 1
w2 = torch.tensor([[-1, 1], [1, 2], [3, 1]])
b2 = -1

h_preact = torch.matmul(x, w1) + b1
h = torch.nn.functional.relu(h_preact)
#preact -> relu를 적용
y = torch.matmul(h, w2) + b2

import torch
input = torch.tensor([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
m = torch.nn.LayerNorm(input.shape[-1])
output = m(input)

m = torch.nn.Dropout(p=0.2)
input = torch.randn(1, 10)
output = m(input)


