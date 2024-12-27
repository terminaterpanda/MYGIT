import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

#hyperparameter define
device = torch.device('mps')
batch_size = 50
epoch_num = 15
learning_rate = 1e-3

train_data = datasets.MNIST(root='./data', train = True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train=False, transform=transforms.ToTensor())
#root = mnistdata 저장할 물리적 공간 위치/train = 데이터를 학습용으로 사용할 것인지 지정 / download = 이미 저장된 데이터가 있다면 false.
#transform = 데이터를 저장하는 동시에 tensor로 변환시키는 옵션
print("training_data_length: ", len(train_data))
print("test_data_length: ", len(test_data))

image, label = train_data[0]

plt.imshow(image.squeeze().numpy(), cmap="gray")
#image_squeeze를 사용해서 차원을 축소하거나 늘려서 2차원의 이미지로 변환시키는 function
plt.title('label : %s' % label)
plt.show()

train_loader = DataLoader(dataset = train_data,batch_size = batch_size,shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 64 channels * 12x12 feature maps (after pooling)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Max pooling and dropout
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        
        # Flatten feature maps into a single vector
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        # Output layer with log-softmax activation
        output = F.log_softmax(self.fc2(x), dim=1)
        return output
    
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

print(model)

model.train()
i = 0
for epoch in range(epoch_num):
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("Train Step: {}\tloss: {:.3f}".format(i, loss.item()))
            i += 1
            
model.eval()
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()
    
print("Test set: Accuracy: {:.2f}%".format(100 * correct / len(test_loader.dataset)))
