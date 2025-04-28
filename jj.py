
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from kobert_transformers import get_kobert_model, get_kobert_tokenizer
from konlpy.tag import Okt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# 데이터 로드
df = pd.read_csv("파일 위치")  # 뉴스 데이터 위치에 따라 괄호 안 내용 변경
df = df[["text", "keywords"]]  # 전처리 과정에 따라 코드 변경 필요

# KoBERT 토크나이저 로드
tokenizer = get_kobert_tokenizer() #KoBERT에서 사용하는 토크나이저

def mask_keywords(text, keywords):
    tokens = tokenizer.tokenize(text)
    keyword_tokens = [tokenizer.tokenize(kw) for kw in keywords.split(',')]

    for kw in keyword_tokens:
        for t in kw:
            if t in tokens:
                tokens[tokens.index(t)] = '[MASK]' # 뉴스 기사에서 핵심 키워드를 [MASK] 토큰으로 처리

    return tokenizer.convert_tokens_to_string(tokens)
df["masked_text"] = df.apply(lambda x: mask_keywords(x["text"], x["keywords"]), axis=1)

# 여기서부터는 모델이 [MASK] 토큰을 예측하면서 fine-tuning 됨, 즉 KoBERT 모델이 뉴스 기사에서 토픽을 추출하는 데 특화되도록 fine-tuning 하는 과정
# 원래 뉴스 키워드를 정답(Label)로 설정하여 학습
class MaskedNewsDataset(Dataset):
    def __init__(self, texts, masked_texts, tokenizer, max_len=256):
        self.texts = texts
        self.masked_texts = masked_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        masked_text = self.masked_texts[idx]

        encoding = self.tokenizer(
            masked_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        label_encoding = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_encoding['input_ids'].squeeze(0)
        }

train_dataset = MaskedNewsDataset(df["text"].tolist(), df["masked_text"].tolist(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#BERTForMasedLM을 사용하여 KoBERT를 fine-tuning
model = BertForMaskedLM.from_pretrained("skt/kobert-base-v1")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# 이제 fine-tuning이 끝난 모델로 새로운 뉴스 데이터에서 키워드를 추출
# 예를 들어 "KoBERT-준표"라는 fine-tuning 된 모델이 새로운 뉴스 본문=데이터를 입력으로 받아 키워드를 [MASK] 토큰으로 변경하게됨, 이 토큰이 뉴스의 키워드임

def extract_keywords(text):
    encoding = tokenizer(
        text, padding='max_length', truncation=True, max_length=256, return_tensors="pt"
    )
    input_ids = encoding["input_ids"]

    with torch.no_grad():
        output = model(**encoding)
        predicted_ids = torch.argmax(output.logits, dim=-1)

    predicted_text = tokenizer.decode(predicted_ids[0])

    # [MASK]가 있던 위치의 단어를 키워드로 추출
    masked_indices = (input_ids == tokenizer.convert_tokens_to_ids("[MASK]")).nonzero()
    keywords = [predicted_text[i] for i in masked_indices]

    return keywords

df["predicted_keywords"] = df["text"].apply(extract_keywords)

# 이제 LDA 알고리즘을 활용하여 주요 키워드 기반 5개의 토픽을 생성, 각 토픽에서 상위 10개 키워드를 확인
# LDA는 이름은 같지만 두 가지 알고리즘이 있음 (Linear Discriminant Analysis=선형판별분석, Latent Drichlet Allocation=잠재 디리클레 할당) 둘 중 원하는 알고리즘을 사용할 것
# 지금의 경우는 잠재 디리클레 할당임

vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["predicted_keywords"].astype(str))

lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 토픽 개수에 따라 변경 필요
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
topics = {f"Topic {i}": [feature_names[idx] for idx in topic.argsort()[:-10 - 1:-1]]
          for i, topic in enumerate(lda.components_)}

print(topics)

# 토픽 클러스터링 및 시각화
# 토픽별 키워드 시각화
wordcloud = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                      background_color="white",
                      width=800, height=600)  # 색상, 크기 변경 필수

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, ax in enumerate(axes.flatten()):
    wc = wordcloud.generate(" ".join(topics[f"Topic {i}"]))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Topic {i}")

plt.tight_layout()
plt.show()
