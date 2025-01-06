import os 
import shutil

original_dataset_dir = "./dataset"
#원본 dataset 저장된 directory 경로

if not os.path.exists(original_dataset_dir):
    os.makedirs(original_dataset_dir)
    
    #os.makedirs(존재하지 않는 디렉토리를 생성할 때 use하는 함수)
    #os.rmdir() -> (디렉토리 삭제)
    #os.system() -> 시스템 명령어 실행
classes_list = os.listdir(original_dataset_dir)
#원본 dataset directory 안에 있는 모든 클래스 이름을 리스트로 가져옴

base_dir = "./splited"
#data 분리하여 저장할 디렉토리 경로(splited라는 이름으로 새 디렉토리를 생성)
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
else:
    print(f"{base_dir}가 이미 존재")

train_dir = os.path.join(base_dir, "train") #디렉토리 경로 생성
if not os.path.exists(train_dir):
    os.mkdir(train_dir) #디렉토리 만듬
validation_dir = os.path.join(base_dir, "val")
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir) 
test_dir = os.path.join(base_dir, "test")
if not os.path.exists(test_dir):
    os.mkdir(test_dir) 

#각 클래스 이름에 대해 반복문을 수행
for clss in classes_list:
    train_class_dir = os.path.join(train_dir, clss)
    if not os.path.exists(train_class_dir):
        os.mkdir(train_class_dir)
        print(f"{train_class_dir} 디렉토리를 생성했습니다.")
    else:
        print(f"{train_class_dir} 디렉토리가 이미 존재합니다.")

    validation_class_dir = os.path.join(validation_dir, clss)
    if not os.path.exists(validation_class_dir):
        os.mkdir(validation_class_dir)
        print(f"{validation_class_dir} 디렉토리를 생성했습니다.")
    else:
        print(f"{validation_class_dir} 디렉토리가 이미 존재합니다.")

    test_class_dir = os.path.join(test_dir, clss)
    if not os.path.exists(test_class_dir):
        os.mkdir(test_class_dir)
        print(f"{test_class_dir} 디렉토리를 생성했습니다.")
    else:
        print(f"{test_class_dir} 디렉토리가 이미 존재합니다.")
"""
디렉토리 구조
splited/
--train
-----class
--val
-----class
--test
-----class
"""  

import math

for clss in classes_list:
    path = os.path.join(original_dataset_dir, clss)
    fnames = os.listdir(path)
    
    train_size = math.floor(len(fnames) * 0.6)
    validation_size = math.floor(len(fnames) * 0.2)
    test_size = math.floor(len(fnames) * 0.2)
    
    train_fnames = fnames[:train_size]
    print(f"train_size({clss}):", len(train_fnames))
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(train_dir, clss), fname)
        shutil.copyfile(src, dst)
        
    validation_fnames = fnames[train_size:(validation_size + train_size)]
    print("validation_size(',clss,'): ", len(validation_fnames))
    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(validation_dir, clss), fname)
        shutil.copyfile(src, dst)
        
    test_fnames = fnames[(train_size + validation_size):
        (validation_size + train_size + test_size)]
    print("test_size(',clss,'): ", len(test_fnames))
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(test_dir, clss), fname)
        shutil.copyfile(src, dst)
        
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 256
EPOCH = 30
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

transform_base = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = ImageFolder(root="./splitted/train", transform=transform_base)
val_dataset = ImageFolder(root="./splitted/val", transform=transform_base)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

class Net(nn.Module):
    #딥러닝 모델과 관련된 기본적인 함수 포함하는 nn.Module 클래스를 상속하여 사용(여러 메서드를 사용할 수 있음)
    
    def __init__(self):
        #init__ 함수에서 모델에서 사용할 모든 layer 정의
        super(Net, self).__init__() #nn.Module 내에 있는 메서드를 상속받아서 use.
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) #(입력수, 출력 수, 커널 크기)
        self.pool = nn.MaxPool2d(2, 2) #커널 크기, stride
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 동적 크기 확인 후 64*8*8으로 수정
        self.fc2 = nn.Linear(512, 33)

    def forward(self, x): #모델이 학습 data를 입력 받아 순전파로 output을 계산하는 과정을 정의함
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
model_base = Net().to(DEVICE)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)

print(model_base)

def train(model, train_loader, optimizer):
    model.train() #입력받는 model을 학습 모드로 설정
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad() #optimizer 초기화
        output = model(data)
        loss = F.cross_entropy(output, target)
        # -> 계산한 loss값을 바탕으로 역전파를 통해 계산한 gradient값을 각 파라미터에 할당
        loss.backward()
        optimizer.step()
        
def evaluate(model, test_loader):
    model.eval() #입력받는 model을 평가모드로 설정
    test_loss = 0#test_loss를 선언, 0으로 초기화
    correct = 0 #올바르게 예측한 data의 수를 세는 변수인 correct를 선언하고, 0으로 초기화
    
    with torch.no_grad(): #모델을 평가하는 동중에는 parameter 업데이트를 하면 안되므로 해당 부분을 실행하는 동안 모델의 parameter update 중단
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            #test_loss == 모델에서 계산한 output값인 예측값과 target값 사이의 loss 계산
            pred = output.max(1, keepdim = True)[1]
            #모델에 입력된 testdata가 33개의 클래스에 속할 확률값이 output으로 출력
            #가장 높은 값을 [1]을 사용해서 예측값으로 저장.
            correct += pred.eq(target.view_as(pred)).sum().item()
            #target.view_as(pred) target_tensor 구조를 pred_tensor과 같은 모양으로 정렬
            #pred.eq == "객체간의 비교 연산자"(일치하지 않으면 0, 일치하면 1을 반환)
    test_loss /= len(test_loader.dataset)#모든 미니 배치의 정확도 값을 batch개수로 나누어서 mean 반환
    test_accuracy = 100 * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

import time 
import copy

def train_baseline(model, train_loader, val_loader, optimizer, num_epochs = 30):
    best_acc = 0.0 #best_acc == 정확도가 가장 높은 모델의 정확도를 저장하는 변수
    best_model_wts = copy.deepcopy(model.state_dict())
    #best_model_wts == 정확도가 가장 높은 모델을 저장할 변수
    
    for epoch in range(1, num_epochs + 1):
        since = time.time() #time.time()을 가지고 와서 해당 epoch가 시작할 때의 시각을 저장
        train(model, train_loader, optimizer) #train으로 모델을 학습
        train_loss, train_acc = evaluate(model, train_loader)
        # evaluate 함수를 사용해서 해당 epoch의 학습 loss와 정확도를 계산
        val_loss, val_acc = evaluate(model, val_loader)
        # 똑같이 함수를 사용해서 검증 loss와 정확도를 계산
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since #소요된 시간을 계산
        #해당 epoch의 검증 정확도가 최고 정확도보다 높다면 검증 정확도 업데이트, 저장
        print("--------------------- epoch {} ------------------".format(epoch))
        
        print("train Loss: {:4f}, Accuracy: {:.2f}%"
              .format(train_loss, train_acc))
        print("completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model
base = train_baseline(model_base, train_loader, val_loader, optimizer, EPOCH
                      )
torch.save(base.state_dict(), "baseline.pt")


data_transforms = {
    "train" : transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        
    ])
}
        
    
        
        