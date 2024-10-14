from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from django import forms
from konlpy.tag import Okt
import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from transformers import BertTokenizer, BertForSequenceClassification

okt = Okt()

# 사전 훈련된 BERT 모델 불러오기 (감정 분석용)
tokenizer = BertTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1')

# 사용자가 입력할 폼 정의
class SentimentForm(forms.Form):
    sentence = forms.CharField(label='입력 문장', max_length=200)

def index(request):
    return HttpResponse("이 페이지는 감정 분석 페이지입니다.")

# 문장 전처리 및 형태소 분석
def morph_sentence(sentence):
    if sentence.strip():
        return " ".join(okt.morphs(sentence.strip()))
    return ""

# 감정 분석 수행 함수 (BERT 모델 활용)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach().numpy()[0]
    # 감정 점수 1-10로 변환
    sentiment_score = np.argmax(scores) + 1
    return sentiment_score

# 폼 제출 및 처리
def analyze_sentiment(request):
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            sentence = form.cleaned_data['sentence']
            # 전처리 후 감정 분석 수행
            clean_sentence = morph_sentence(sentence)
            score = predict_sentiment(clean_sentence)
            return HttpResponse(f"문장: {sentence}, 감정 점수: {score}/10")
    else:
        form = SentimentForm()

    return render(request, 'analyze.html', {'form': form})



