import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x -4
from konlpy.tag import Mecab

tokenizer = Mecab()
print(tokenizer.morphs("강방성"))