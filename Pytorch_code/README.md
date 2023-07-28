---
title: Network：神经网络代码
date: 2022-11-25 16:20:08
tags:
- python
- 深度学习
---
## TORCH.NN
**Module 所有神经网络模块的基类**
```py
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module): # 继承Module模板（父类），在该模板基础上进行修改
    def __init__(self): #初始化
        super(Model, self).__init__() 	#父类初始化
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0) #2维卷积
    def forward(self, x):#前向传播 x即input
        x = self.conv1(x)
        return x 
model=Model() 		#创建神经网络
output=model(x)
print(output)
```
**maxpool 下采样**
``self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False) #ceil_mode=True为向上取整``
**Non-linear Activations**
``m = nn.ReLU() # inplace=True为直接替换input``
**批标准化**
``m = nn.BatchNorm2d(100)``
**线性层**
```py
dataset = torchvision.datasets.CIFAR10("../data", train=False，
			transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64)
 
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)
 
    def forward(self, input):
        output = self.linear1(input)
        return output
 
tudui = Tudui()
 
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs) #数据摊平
    print(output.shape)
    output = tudui(output)
    print(output.shape)
```
## CIFAR 10 model结构
![图 1](https://raw.sevencdn.com/Arrowes/Arrowes-Blogbackup/main/images/Network1.png)  

```py
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
 
 
class WYJ(nn.Module):
    def __init__(self):
        super(WYJ,self).__init__()
        self.conv1=Conv2d(3,32,5,padding=2)
        self.maxpool1=MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten=Flatten()
        self.linear1=Linear(1024,64)
   self.linear2 = Linear(64,10)
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
 
wyj = WYJ()
print(wyj)
input = torch.ones((64, 3, 32, 32))
output = wyj(input)
print(output.shape)
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
 # Sequential 的妙用
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
 
    def forward(self, x):
        x = self.model1(x)
        return x
 
tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)
 
writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)
writer.close()
```
## 损失函数
```py
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()    #注意输入输出形状
result_cross = loss_cross(x, y)
```
**反向传播**
``result_loss.backward()``

## 优化器(梯度下降)
```py
for input, target in dataset:
    optimizer.zero_grad()              #梯度清零
    output = model(input)
    loss = loss_fn(ou tput, target)
    loss.backward()
    optimizer.step()
```
## 网络模型
1. 添加
``vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10)) #在vgg16网络中的classifier 模块中加入一个linear层``
2. 修改
``vgg16_false.classifier[6] = nn.Linear(4096, 10) #修改vgg16网络中的classifier 第6层``

3. 保存
```py
vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")
 
# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
```
4. 加载
```py
# 方式1
model = torch.load("vgg16_method1.pth")
 #若为自定义网络模型，加载时需要引入定义：
from model_save import *  
#一定要在同一个文件夹下
 
# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth")) #读取字典中的参数
```
 
## CPU训练
 ```py
import …
from model import * #引入CIFAR 10 model网络定义

 #准备数据集 
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),download=True)

 #length 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
 
 #利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
 
 #创建网络模型
tudui = Tudui()
 #损失函数
loss_fn = nn.CrossEntropyLoss()
 #优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
 
 #设置训练网络的一些参数
 #记录训练的次数
total_train_step = 0
 #记录测试的次数
total_test_step = 0
 #训练的轮数
epoch = 10
 
 #添加tensorboard
writer = SummaryWriter("../logs_train")
 
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))
    #训练步骤开始
    tudui.train() #网络设为训练模式
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
 
        # 优化器优化模型
        optimizer.zero_grad() #梯度清零
        loss.backward()
        optimizer.step()
 
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0: #逢百进一
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
 
    # 验证步骤开始
    tudui.eval() #网络设为验证模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): #避免验证过程加入梯度，以节约内存
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() #argmax(1)：横轴取最大值并输出序号[0,1,0,0,…]
            total_accuracy = total_accuracy + accuracy
 
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    #每epoch保存一个
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存") 
writer.close()
 ```
## GPU训练	
改cuda：网络、损失函数、data
① .cuda()
```py
if torch.cuda.is_available():
    tudui = tudui.cuda()
```
② .to(device)
```py
device = torch.device("cuda") #只需修改此处在CPU,GPU之间切换	“cpu”
tudui = tudui.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else “cpu”)
```
## 测试
```py
model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu')) 
print(model) 	#训练好的权重代入模型
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
```
