import pandas as pd
import numpy as np
import torch as nn
import nltk
import sys
import re

class clean():

    def __init__(self, text):
        self.text = text

    def clean_text(text):
        return text.apply(lambda text: re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)) if isinstance(text, str) else "")


class GET():
    def __init__(self, name, file_path):
        self.name = name
        file_path = "/Users/iseong-yong/Desktop/files/movie1.csv"
        textfile = pd.read_csv(file_path, encoding ="utf-8")
        textfile = textfile.replace(".", "")
        return textfile

nltk.download('stopwords')
from nltk.stem import PorterStemmer




