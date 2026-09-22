---
title: Py：conda，Pycharm，Pytorch
date: 2022-11-24 21:26:44
tags: 
- python
---
深度学习相关工具使用笔记，包括Anaconda, Pycharm, Jupyter notebook, Google Colab，以及Pytorch，项目地址：[DLpractice](https://github.com/Arrowes/DLpractice)
<!--more-->

视频：[PyTorch深度学习快速入门教程](https://www.bilibili.com/video/BV1hE411t7RN/)

# Anaconda
+ Windows环境配置
  安装[miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe), [镜像源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/?C=M&O=D) （或[**Anaconda**](https://www.anaconda.com/)，更大更全但没必要）
  防止环境装在C盘占空间：修改user目录下.condarc文件里的默认地址，或执行``conda config --add D:\Anaconda3\envs ``,然后``conda info`` 检查envs directories
  （若报错 The channel is not accessible or is invalid 运行``conda config --remove-key channels``）

+ Linux 环境配置
  安装miniconda，相较Anaconda更小巧快捷，功能一样
  ```sh
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  #一路Enter + Yes，最后使修改的PATH环境变量生效：
  source ~/.bashrc
  conda   #验证是否成功
  conda create -n pytorch python=3.6  #创建一个名为pytorch的环境
  conda activate pytorch  #激活环境
  ```
  > （若要安装：[Anaconda](https://www.anaconda.com/)，执行下载的.sh文件，输入``bash XXX.sh``，然后一路enter和yes；激活：``cd ///root/anaconda3/bin``,输入：``source ./activate``，终端前出现``(base)``则激活成功）

```sh
conda create -n pytorch python=3.8  #Anaconda创建环境
conda update python #更新py
conda activate pytorch  #激活环境
conda deactivate    #退出虚拟环境
conda remove pytorch --all  #删除环境 或conda env remove -n XXX
conda list          #查看环境中的所有包
conda install XXX   #安装 XXX 包
conda remove XXX    #删除 XXX 包
conda env list      #列出所有环境
conda create -n XXX jupyter notebook  #创建环境并安装Jupyter Notebook
jupyter notebook    #打开Jupyter Notebook

pip install XXX==2.0 (pip uninstall XXX) #安装特定版本/卸载
pip install --upgrade pip #更新pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  #pip换源
pip config list -v  #查源
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/  #pip换源安装
conda config --remove-key channels    #换回默认源
```
> 国内镜像源:
  清华大学：https://pypi.tuna.tsinghua.edu.cn/simple
  阿里云：http://mirrors.aliyun.com/pypi/simple
  豆瓣：http://pypi.douban.com/simple

vscode导入conda：当在VScode中打开某py文件时，左下角会出现Python, 点击可切换conda环境


# Pycharm
创建新项目，手动导入已存在的anaconda创建的环境``D:\Anaconda3\envs\pytorch\python.exe``

配置或更改环境：				``Settings > Project > interpreter > 齿轮 > Add > System Interpreter > 手动添加``

Terminal：``File > Settings > Tools > Terminal > Shell path`` ``C:\Windows\System32\cmd.exe``

**Pycharm使用**
> 新建.py文件(用作工程) > 右上角配置py解释器 > 运行/直接右键运行
控制台(用作调试，查参数) > Shift+enter：输入多行 > “↑”重新编辑
Ctrl + D 快速复制至下一行
Ctrl + R 批量修改
Ctrl + / 批量注释
调试：打断点 > debug > 使用工具栏内的“下一步”或“计算器内输入表达式”进行调试

调试时使用Console的python调试台，print指令
![图 1](https://raw.gitmirror.com/Arrowes/Blog/main/images/Pytorch1.png) ![图 2](https://raw.gitmirror.com/Arrowes/Blog/main/images/Pytorch2.png)  


```py
#Debug:
ModuleNotFoundError: No module named 'xxx'
#需要导入项目根目录：
import sys
sys.path.append('/home/ywang85/edgeai-yolox')
```

# Jupyter notebook
激活pytorch环境后: ``conda install nb_conda``
打开 jupyter notebook: ``New > 选择环境：Python [conda env:pytorch]``
IDLE Ctrl+N 编辑多行代码
输入一半按tab可以补全

打包与解压缩
+ 打包：
  ```py
  import zipfile

  zip_ref = zipfile.ZipFile('WYJ.zip', 'r')
  zip_ref.extractall('.')
  zip_ref.close()
  ```

+ 解压：
  ```py
  import zipfile
  import os

  def zip_folder(folder_path, zip_path):
      with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
          for root, dirs, files in os.walk(folder_path):
              for file in files:
                  file_path = os.path.join(root, file)
                  zipf.write(file_path, os.path.relpath(file_path, folder_path))

  folder_path = './output' # 指定要下载的文件夹路径
  zip_path = './output.zip' # 指定要保存的zip文件路径
  zip_folder(folder_path, zip_path)

  from IPython.display import FileLink
  FileLink(zip_path) # 生成下载链接
  ```



# [Google Colab](https://colab.research.google.com/)

```py
#设置并查看GPU 修改>笔记本设置>GPU
import tensorflow as tf
tf.test.gpu_device_name()

!/opt/bin/nvidia-smi #详情
```

**基本指令**
```py
!unzip /content/XX.zip -d /content/XX #解压
%cd /content/XX   #进入
!pip install -r requirements.txt    #安装requirements
!python XX.py --rect    #运行
!rm -rf /content/XX/mydata  #删除

%load_ext tensorboard   #加载tensorboard
%tensorboard --logdir=runs/train    #执行tensorboard
```

**云盘**
```py
#先装载谷歌云盘，在云盘里运行以防数据丢失，指定Google Drive云端硬盘的根目录，名为drive
!mkdir -p drive
!google-drive-ocamlfuse drive

#connect to self drive
from google.colab import drive
drive.mount('/content/drive')
#云训练时还是要将盘里的文件拿出来再开始，否则容易直接断连!
```
续航插件：[Colab Alive](https://chrome.google.com/webstore/detail/colab-alive/eookkckfbbgnhdgcbfbicoahejkdoele?hl=zh-CN), 防止训练时掉线

# Pytorch
要调用GPU进行训练的话，需要安装显卡驱动对应的CUDA
1. ``nvidia-smi`` 查询支持CUDA版本, 显卡驱动程序显示的cuda版本为电脑最高可适配的cuda版本
![图 3](https://raw.gitmirror.com/Arrowes/Blog/main/images/Pytorch3.png)  
2. 到 [Pytorch官网](https://pytorch.org/get-started/locally/) 复制对应code进行安装
![图 4](https://raw.gitmirror.com/Arrowes/Blog/main/images/Pytorch4.png)  

## 安装
**windows**
安装显卡驱动对应的**CUDA**：``nvidia-smi`` 查询支持CUDA版本，
再到[Pytorch官网](https://pytorch.org/get-started/locally/)复制对应code进行安装, 如：
``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia``
（验证torch能否使用GPU：`python -c "import torch;print(torch.cuda.is_available())"`   返回True说明GPU可以被使用）

**Linux**
+ pytorch
``conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``
或`pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`

+ tensorflow
cuda:``conda install cudatoolkit=10.0``
cuDNN:``conda install cudnn=7.6``
tf:``pip install tensorflow-gpu==1.15.0``(注意版本匹配)

**查GPU**
```py
python -c "import torch;print(torch.cuda.is_available())"   #返回True说明GPU可以被使用
torch.__version__ #查pytorch版本
nvidia-smi -l 2   #查GPU CUDA, '-l 2':每2s更新一次
python –-version  #查python版本
conda install python=3.8  #升级(覆盖安装)python
```

## 库

**两大查询函数**：
dir() 函数，能让我们知道工具箱以及工具箱中的分隔区有什么东西。
help() 函数，能让我们知道每个工具是如何使用的，工具的使用方法。
``Jupyter>XX??``
``Pycharm>ctrl+左键(查原函数)	ctrl+p(查询输入参数，有等号的可忽略)``
多查 [pytorch官方文档](https://pytorch.org/docs/stable/index.html)

+ 文件
  ```py
  ../XXX #上一层
  root=“D:\\desktop”  #window下绝对路径使用双斜杠\\避免转义：
  root=r“D:\\desktop” #或统一加上r取消转义
  ```
+ 计时		
  ```py
  import time 	
  start=time.time()     end=time.time()     print(start-end)
  ```
+ 组 
  ```py
  a = (1, 2)    # 元组 tuple
  b = [1, 2, 3] # 数组 list
  c = {'name': 'wyj', 'age': '23'}  # 字典 dict

  print(a[0])
  print(b[1])
  print(c['name'])
  ```

### Tensorboard可视化
pytorch下安装 ``pip install tensorboard (conda install tensorboard)``

+ 使用
  ```py
  from torch.utils.tensorboard import SummaryWriter
  writer=SummaryWriter(“logs“)
  writer.add_image("name“，parameter，组内步数)
  writer.close() #关闭读写 
  ```
+ 打开
  ```py
  tensorboard --logdir=logs(文件夹路径) --port=6006（6007） #注意路径
  tensorboard --logdir runs/train  （YOLO）
  ```
  地址	localhost:6006
  ```py
  #debug
  AttributeError: type object 'h5py.h5.H5PYConfig' has no attribute '__reduce_cython__'
  pip uninstall h5py

  AttributeError: module 'distutils' has no attribute 'version'
  pip install setuptools==59.5.0
  pip install brotli
  ```
### Transform
transforms.py图像处理工具箱
1. 调用工具tool=transforms.XXX()	
2. 使用 result=tool(input)
如: Totensor>转向量; Normalize>归一化; Resize>缩放; Compose>组合; RandomCrop>随机裁剪

### ToTensor
```py
#（桥梁，很多输入都要求tensor类型）
from torchvision import transforms
tensor_tool=transforms.ToTensor()
tensor_result=tensor_tool(img)
```

```py
class Person:
	_ _call_ _(self,name)	
可直接调用>person=Person(“wyj")
```
### torchvision
torchvision.datasets 数据集处理

```py
import torchvision
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, #训练集
transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, #测试集
transform=dataset_transform, download=True)
```
### Dataloader
```py
from torch.utils.data import DataLoader
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
```
例
```py
writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader: #读数据
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1
writer.close()
```

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
![图 1](https://raw.gitmirror.com/Arrowes/Blog/main/images/Network1.png)  

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
 
writer = SummaryWriter("../logs_seq")
writer.add_graph(wyj, input)
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