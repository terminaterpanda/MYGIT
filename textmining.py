#####################################################################
import pandas as pd
import re
from collections import Counter
import numpy as np
from konlpy.tag import Okt
 
# 이후 형태소 분석 등의 작업을 수행

file_path7 = "/Users/iseong-yong/Desktop/Rfolder/speech_moon.txt"
file_path8 = "/Users/iseong-yong/Desktop/Rfolder/speech_yoon.txt" 

moon = pd.read_table(file_path7, encoding="utf-8-sig")
yoon = pd.read_table(file_path8, encoding="utf-8")

#moon = 1*57
#yoon = 1*86

m1 = " ".join(moon.astype(str))
y1 = " ".join(yoon.astype(str))

m1 = m1.split(".")
y1 = y1.split(".")

#온점을 기준으로 문장 쪼개기 완료.



m1_freq = pd.Series(m1).value_counts()
y1_freq = pd.Series(y1).value_counts()

m2 = m1_freq.reset_index()
y2 = y1_freq.reset_index()

m2.columns = ["nouns", "frequency"]
y2.columns = ["nouns", "frequency"]

filepath1 = "/Users/iseong-yong/Desktop/files/m1"
filepath2 = "/Users/iseong-yong/Desktop/files/y1"

m2.to_csv(filepath1, index=False, encoding="utf-8-sig")
y2.to_csv(filepath2, index=False, encoding="utf-8-sig")





