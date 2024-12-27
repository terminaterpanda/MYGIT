#문맥을 반영한 워드 임베딩
#contextualized word embedding
large_a = list(range(1000))
large_b = list(range(1000))

import numpy as np

def numpy_dotproduct_approach(x, w):
    #same as np.dot
    return x.dot(w)

a = np.array([1., 2., 3.])
b = np.array([4., 5., 6.])

print(numpy_dotproduct_approach(a,b))

large_a = np.arange(1000)
large_b = np.arange(1000)

a = [1., 2., 3.]
np.array(a)


lst = [[1,2,3],
       [4,5,6]]
ary2d = np.array(lst)
ary2d

#row*column

ary2d.dtype

int32_ary = ary2d.astype(np.int32)
int32_ary

float_ary = ary2d.astype(np.float32)
float_ary
#ELMO 
#1. 각 층의 출력값을 concatenate
#2. 출력값별로 가중치 부여
#3. 출력값을 전부 더함
#4. 스칼라 매개변수 곱
import tensorflow_hub as hub
#tensorflow-hub == "다양한 사전 훈련 모델 use"
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# 텐서플로우 허브로부터 ELMo를 다운로드

sess = tf.Session()
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin-1')
print(data[:5])
