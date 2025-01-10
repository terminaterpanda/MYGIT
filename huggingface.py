from Korpora import Korpora
nsmc = Korpora.load("nsmc", force_download=True)
#data 내려받기


#순수 text형태로 저장하기
import os
def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
write_lines("/MYGIT/train.txt", nsmc.train.get_all_texts())
write_lines("/MYGIT/test.txt", nsmc.test.get_all_texts())

import os
os.makedirs("/MYGIT/staree", exist_ok=True)

from tokenizers import ByteLevelBPETokenizer
#textcnn 모델링

import torch.nn as nn
import torch.optim as optim
#optim에서 model 최적화 알고리즘 가져오기
import torch.nn.functional as F #활성함수 등등을 가져오는 module
import torch
class TextCNN(nn.Module):
    #nn.module에서 레이어 및 함수를 가져옴
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
    #textCNN -> 객체 초기화    
        super(TextCNN, self).__init__()
        #nn모듈 내에 있는 메서드를 상속받아서 이용
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim))
                                    for w in kernel_wins])
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        #클래스에 대한 score를 생성하기 위해서 
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)
        
        con_x = [self.relu(conv(emb_x)) for conv in self.convs]
        
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2])for x in con_x]
        
        fc_x = torch.cat(pool_x, dim =1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x) 
        
        logit = self.fc(fc_x)
        
        return logit
    def train(model, device, train_itr, optimizer):
        model.train()
        #textcnn model을 train_mode로 변경하여 parameter 업데이트 가능케 함
        corrects, train_loss = 0.0, 0
        #corrects, train_loss 둘 다 0으로 설정하여 계산 가능케 함
        for batch in train_itr: #
            text, target = batch.text, batch.label
            #미니 배치 단위로 저장된 텍스트와 label 데이터를 저장
            text = torch.transpose(text, 0, 1)
            #역행렬로 반환
            target.data.sub_(1)
            #target 값을 1씩 줄임
            text, target = text.to(device), target.to(device)
            #장비에 할당
            optimizer.zero_grad()
            #optimizer를 다시 초기화
            logit = model(text)
            #logit == "text 데이터를 input으로 사용하여 output 계산"
            loss = F.cross_entropy(logit, target)
            #softmax를 통과시켜서 yes, no로 분류
            loss.backward() #미분하면서 누적된 gradient 계산
            #parameter 값 update
            optimizer.step()
            
            train_loss += loss.item()
            result = torch.max(logit, 1)[1]
            corrects += (result.view(target.size()).data == target.data).sum()
            
        train_loss /= len(train_itr.dataset)
        accuracy = 100.0 *corrects / len(train_itr.dataset)
        
        return train_loss, accuracy
    
    def evaluate(model, device, itr):
        model.eval()
        corrects, test_loss = 0.0, 0
        
        for batch in itr:
            
            text = batch.text
            target = batch.label
            text = torch.transpose(text, 0, 1)
            target.data.sub_(1)
            text, target = text.to(device), target.to(device)
            
            logit = model.text()
            loss = F.cross_entropy(logit, target)
            
            test_loss += loss.item()
            result = torch.max(logit, 1)[1]
            corrects += (result.view(target.size()).data == target.data).sum()
            
        test_loss /= len(itr.dataset)
        accuracy = 100.0 * corrects / len(itr.dataset)
        
        return test_loss, accuracy
    
