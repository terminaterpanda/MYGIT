import pandas as pd
import numpy as np
import re

def clean_text(text_series):
    # Series의 각 요소에 대해 정제 작업 수행
    return text_series.apply(lambda text: re.sub(r"[^가-힣a-zA-Z\s]", "", str(text)) if isinstance(text, str) else "")

# CSV 파일 경로 설정
file_path1 = "/Users/iseong-yong/Desktop/files/movie1.csv"
textfile = pd.read_csv(file_path1, encoding="utf-8")

# 텍스트에서 마침표 제거
textfile = textfile.replace(".", "")

# 평점에 따라 데이터 분리
textfile_1 = textfile[textfile.point < 5]  # 평점이 5점 미만인 데이터
textfile_2 = textfile[textfile.point >= 5]  # 평점이 5점 이상인 데이터

# 각 데이터에서 특정 열(예: 리뷰 텍스트) 추출 후 인덱스 재설정
t1 = textfile_1.iloc[:, 2].reset_index(drop=True)
t2 = textfile_2.iloc[:, 2].reset_index(drop=True)

# 텍스트 정제
text1 = clean_text(t1)
text2 = clean_text(t2)

# 데이터프레임 생성
text1_df = pd.DataFrame(text1, columns=['reviews'])  # 평점 5점 미만
text2_df = pd.DataFrame(text2, columns=['reviews'])  # 평점 5점 이상

# CSV 파일로 저장
f_path = "/Users/iseong-yong/Desktop/files/negativedata.csv"  # 부정적 데이터 파일 경로
f_path2 = "/Users/iseong-yong/Desktop/files/positivedata.csv"  # 긍정적 데이터 파일 경로

text1_df.to_csv(f_path, index=False, encoding="utf-8-sig")  # 부정적 데이터 저장
text2_df.to_csv(f_path2, index=False, encoding="utf-8-sig")  # 긍정적 데이터 저장

