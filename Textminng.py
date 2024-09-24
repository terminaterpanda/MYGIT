import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer

file_path1 = "/Users/iseong-yong/Desktop/files/movie1.csv"
 
textfile = pd.read_csv(file_path1, encoding= "utf-8")
textfile_1 = textfile[textfile.point < 5]
#580*3 matrix
textfile_2 = textfile[~textfile.index.isin(textfile_1.index)]
#이런 식으로 위의 조건을 제외.
#420*3 matrix
t1 = textfile_1.iloc[0:580,2].reset_index(drop=True)
t2 = textfile_2.iloc[0:420,2].reset_index(drop=True)
#rest_index == 인덱스를 재정렬해주는 function.

okt = Okt()
texter = " ".join(t1.astype(str))
texter2 = " ".join(t2.astype(str))

nouns = okt.nouns(texter)
nouns1 = okt.nouns(texter2)
nouns_freq = pd.Series(nouns).value_counts()
nouns_freq1 = pd.Series(nouns1).value_counts()
#value_counts == 기본적으로 지정된 열의 값들에 대한 모든 밞생 횟수를 반환.
nouns_df = nouns_freq.reset_index()
nouns_df2 = nouns_freq1.reset_index()
nouns_df.columns = ["noun", "frequency"]
nouns_df2.columns = ["noun", "frequency"]

print(nouns_df)
print(nouns_df2)

output_file_path = "/Users/iseong-yong/Desktop/files/nouns_frequency_negative.csv"
output_file_path1 = "/Users/iseong-yong/Desktop/files/nouns_frequency_positive.csv"

nouns_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
nouns_df2.to_csv(output_file_path1, index=False, encoding="utf-8-sig")

print(nouns_df.shape)
print(nouns_df2.shape)

