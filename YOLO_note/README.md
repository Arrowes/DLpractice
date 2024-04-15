---
title: Yolo：Code，Config，Ideas
date: 2022-11-27 16:04:02
tags:
- python
- 深度学习
---
记录了YOLO的环境配置、资料代码、魔改记录、炼丹经验、论文想法。
<!--more-->

# 环境配置
详细使用见 [Anaconda，Pycharm，Jupyter，Pytorch](https://wangyujie.space/Pytorch/)

**Windows环境配置**
1. 安装[miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe), [镜像源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/?C=M&O=D) （或[**Anaconda**](https://www.anaconda.com/)，更大更全但没必要）
防止环境装在C盘占空间：修改user目录下.condarc文件里的默认地址，或执行``conda config --add D:\Anaconda3\envs ``,然后``conda info`` 检查envs directories
（若报错 The channel is not accessible or is invalid 运行``conda config --remove-key channels``）

2. **配置环境**：打开Anaconda Prompt
创建环境``conda create -n pytorch python=3.8``
激活环境``conda activate pytorch``

3. 安装显卡驱动对应的**CUDA**：``nvidia-smi`` 查询支持CUDA版本，
再到[Pytorch官网](https://pytorch.org/get-started/locally/)复制对应code进行安装, 如：
``conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia``
（验证torch能否使用GPU：`python -c "import torch;print(torch.cuda.is_available())"`   返回True说明GPU可以被使用）

4. 安装[**Pychram**](https://www.jetbrains.com/pycharm/), 用pycharm打开YOLO项目文件夹，配置编辑器``D:P\Anaconda3\envs\pytorch\python.exe``，在pycharm的terminal中打开pytorch环境

5. 安装各种**包**：``pip install -r requirements.txt``,
换源补装失败的包``pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/``

**Linux 环境配置**
1. 安装miniconda，相较Anaconda更小巧快捷，功能一样
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

2. 下载pycharm，解压，进入bin文件夹，运行``./pycharm.sh``以打开pycharm（更简单且能生成图标的方法：``sudo snap install pycharm-community --classic``）
在项目中导入环境``.conda/envs/pytorch/bin/python3.6``
3. 安装CUDA
    + **pytorch**
    ``conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``
    或`pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`

    + **tensorflow**
    cuda:``conda install cudatoolkit=10.0``
    cuDNN:``conda install cudnn=7.6``
    tf:``pip install tensorflow-gpu==1.15.0``(注意版本匹配)

如果requirements中有包实在安不上，手动装包：进[网站](https://pypi.org/)搜索包，下载.whl，在包所在位置激活环境运行``pip install [].whl``(包名中cp38代表python3.8版本)
vscode导入conda：当在VScode中打开某py文件时，左下角会出现Python, 点击可切换conda环境

# 资料与代码

| Model   |Paper  | Code|
|---------|-------|-------|
| YOLOv1  | [You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)        | [Code](https://pjreddie.com/darknet/yolov1/)        |
| YOLOv2  | [YOLO9000:Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf) | [Code](https://pjreddie.com/darknet/yolo/)|
| YOLOv3  | [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)| [Code](https://github.com/ultralytics/yolov3)|
| YOLOv4  | [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)| [Code](https://github.com/Tianxiaomo/pytorch-YOLOv4)|
| YOLOv5  | /|      [Code](https://github.com/ultralytics/yolov5)|
| YOLOv6  | [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/pdf/2209.02976.pdf)|              [Code](https://github.com/meituan/YOLOv6)|
| YOLOv7  | [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)|   [Code](https://github.com/WongKinYiu/yolov7)|
| YOLOv8  | /|     [Code](https://github.com/ultralytics/ultralytics)  |
|CEAM-YOLOv7| [CEAM-YOLOv7:Improved YOLOv7 Based on Channel Expansion Attention Mechanism for Driver behavior detection](https://ieeexplore.ieee.org/document/9980374/metrics) |       [Code](https://github.com/Arrowes/CEAM-YOLOv7)
|FEY-YOLOv7| [A Driver Fatigue Detection Algorithm Based on Dynamic Tracking of Small Facial Targets Using YOLOv7](https://www.jstage.jst.go.jp/article/transinf/E106.D/11/E106.D_2023EDP7093/_article) | [Code](https://github.com/Arrowes/FEY-YOLOv7)

YOLOv1 - v5历程：[从yolov1至yolov5的进阶之路](https://blog.csdn.net/wjinjie/article/details/107509243)
YOLOv3论文精读视频：[同济子豪兄YOLOV3目标检测](https://www.bilibili.com/video/BV1Vg411V7bJ/?)
YOLOv5知识精讲：[Yolov5核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/172121380)，[YOLOV5-5.x 源码讲解](https://blog.csdn.net/qq_38253797/article/details/119043919)
YOLOv7网络结构：[理解yolov7网络结构](https://blog.csdn.net/athrunsunny/article/details/125951001) ,[Yolov7 基础网络结构详解](https://blog.csdn.net/u010899190/article/details/125883770)
全流程指导视频：[目标检测 YOLOv5 开源代码项目调试与讲解实战-土堆](https://www.bilibili.com/video/BV1tf4y1t7ru/)


算法精品仓库：[Bubbliiiing](https://github.com/bubbliiiing), [YOLO Air](https://github.com/iscyy/yoloair)，[YOLO Air2](https://github.com/iscyy/yoloair2), [yolov5_research](https://github.com/positive666/yolov5_research)

[YOLO_Note](https://github.com/Arrowes/DLpractice/blob/main/DL_note/YOLO_Note.pdf)

# Ideas
## 数据集
> [Kaggle数据集](https://www.kaggle.com/datasets)
[格物钛数据集](https://gas.graviti.cn/open-datasets)
[Roboflow数据集](https://universe.roboflow.com/roboflow-100)
[IEEE DataPort](https://ieee-dataport.org/datasets)

标注工具：[Roboflow](https://app.roboflow.com/395841716-qq-com)

开源驾驶员行为数据集：[StateFarm-distracted-driver-detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

数据增强：抖动模糊；三种不同的数据增强方法合成三通道；针对红外图像优化
扩大数据集：旋转 偏移（首先要保证原始数据量够）；混合数据集——彩色+红外
各集种类分配不均，测试集要用不同的人


## Anchor
设计——anchor的计算函数Autoanchor
![图 1](https://raw.gitmirror.com/Arrowes/Blog/main/images/Yolo1.png)  


## 网络结构
1. 在 `models/common.py` 加入新的结构代码
2. 在`models/yolo.py` 的parse_model函数中引入上面新写的结构名称
3. `.yaml` 修改网络结构
![图 2](https://raw.gitmirror.com/Arrowes/Blog/main/images/Yolo2.png)  

## 注意力模块
[CV中即插即用的注意力模块](https://zhuanlan.zhihu.com/p/330535757)
[手把手带你YOLOv5 (v6.1)添加注意力机制](https://blog.csdn.net/weixin_43694096/article/details/124443059?spm=1001.2014.3001.5502)

> 位置：
在上采样+concat之后接一个注意力机制可能会更好？
backbone结尾使用一个注意力机制？
每个block（如residual block）结尾使用比每个Conv里使用更好？

transformer自注意力模块 CBAM注意力模块 CA注意力模块 SE注意力模块

## 激活函数 activations.py
[改进激活函数为ReLU、RReLU、Hardtanh、ReLU6、Sigmoid、Tanh、Mish、Hardswish、ELU、CELU等](https://blog.csdn.net/m0_70388905/article/details/128753641)
> activations.py：激活函数代码写在了activations.py文件里，可引入新的激活函数
common.py：替换激活函数，很多卷积组都涉及到了激活函数（Conv，BottleneckCSP），所以改的时候要全面

例：插入激活函数：Mish
1.在utils/activation.py中定义Mish激活函数
2.重构Conv模块，改激活函数：
```py
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.Mish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool
    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        return F.relu(x, inplace=self.inplace)
    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
```

## Loss Function
例：改 EIOU loss
1. 修改 general.py，增加EIOU。
```py
elif EIoU:
                w=(w1-w2)*(w1-w2)
                h=(h1-h2)*(h1-h2)
                return iou-(rho2/c2+w/(cw**2)+h/(ch**2))#EIOU  2021.12.29
```
2. 将loss.py中边框位置回归损失函数改为eiou。
```py
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, EIoU=True)  # iou(prediction, target)
```

## 参数配置（YOLOv5）
**Detect参数**
调用电脑摄像头: 
右上角py配置 > Edit Configurations > Parameters
``--view-img --source 0``

调用手机摄像头：
下载 [IP摄像头](https://www.123pan.com/s/goS7Vv-QeKbd.html) App，关闭代理，连同一个网，Parameters配置为：
``--source http://admin:admin@192.168.43.1:8081`` 具体地址见 APP

**Train参数**
`action='store_true'` 触发了为true，否则为false 和 default=False 效果一样

**YOLOv8（没搞懂）**
该版本参数集中配置ultralytics/yolo/configs/default.yaml
model参数可以是pt也可以是yaml。
>pt:相当于使用预训练权重进行训练，比如选择为yolov8n.pt，就是训练一个yolov8n模型，并且训练前导入这个pt的权重。
yaml:相当于直接初始化一个模型进行训练，比如选择为yolov8n.yaml，就是训练一个yolov8n模型，权重是随机初始化。

data.yaml数据只能用绝对地址
要修改代码先卸ultralytics包，利用setup.py

# 炼丹经验
+ **数据集**：输入图像的大小要求必须是32的倍数；Resize保持原始图像比例调整大小更安全；标注时标注框的设计影响精度

+ **配置**：mosic有时没用可删；删卷积层可减少计算量；若显存不足需要调小batchsize或数据集分辨率；可以从小模型中学到的权重开始，对更大模型进行训练
    + 大的batch_size往往建议可以相应取大点learning_rate, 因为梯度震荡小，大learning_rate可以加速收敛过程，也可以防止陷入到局部最小值，而小batch_size用小learning_rate迭代，防止错过最优点，一直上下震荡没法收敛
    + 参数调优过程一般要反复多次进行`微调<—>训练<—>测试`，最终得出符合需求/较优的HyperPara，应用在项目中	`data/hyps/hyp.finetune.yaml`

**小目标检测**：小目标检测效果不好主要原因为小目标尺寸问题。
以网络的输入608×608为例，yolov5中下采样使用了5次，因此最后的特征图大小是19×19，38×38，76×76。三个特征图中，最大的76×76负责检测小目标，而对应到608×608上，每格特征图的感受野是608/76=8×8大小。即如果原始图像中目标的宽或高小于8像素，网络很难学习到目标的特征信息。
另外很多图像分辨率很大，如果简单的进行下采样，下采样的倍数太大，容易丢失数据信息。但是倍数太小，网络前向传播需要在内存中保存大量的特征图，极大耗尽GPU资源,很容易发生显存爆炸，无法正常的训练及推理。
这种情况可以使用**分割**的方式，将大图先分割成小图，再对每个小图检测，不过这样方式有优点也有缺点： 
> 优点：准确性 分割后的小图，再输入目标检测网络中，对于最小目标像素的下限会大大降低。
比如分割成608×608大小，送入输入图像大小608×608的网络中，按照上面的计算方式，原始图片上，长宽大于8个像素的小目标都可以学习到特征。
缺点：增加计算量 比如原本1920×1080的图像，如果使用直接大图检测的方式，一次即可检测完。但采用分割的方式，切分成4张912×608大小的图像，再进行N次检测，会大大增加检测时间。
[YOLOV5 模型和代码修改——针对小目标识别-CSDN博客](https://blog.csdn.net/weixin_56184890/article/details/119840555)

此外，也可以增加一个小目标检测层：[增加小目标检测层-CSDN博客](https://blog.csdn.net/m0_70388905/article/details/125392908)

使用云服务器快速训练（收费）：[AutoDL算力云 | 弹性、好用、省钱](https://www.autodl.com/home)
[AutoDL帮助文档-GPU选型](https://www.autodl.com/docs/gpu/)

# 其他未实现的想法
剪枝：[模型剪枝、蒸馏、压缩-CSDN博客](https://blog.csdn.net/m0_70388905/article/details/128222629)
[GitHub - Torch-Pruning: [CVPR 2023] Towards Any Structural Pruning; ](https://github.com/VainF/Torch-Pruning)

融合EfficientNet和YoloV5：主要思想是训练一个图像分类模型(EfficientNet)，它可以实现非常高的AUC(约0.99)，并找到一种方法将其与目标检测模型融合。这被称为“2 class filter”

加权框融合(WBF)后处理：对目标检测模型产生的框进行过滤，从而使结果更加准确和正确的技术。它的性能超过了现有的类似方法，如NMS和soft-NMS。

用5折交叉验证
双流网络
矩形训练
PERCLOS值怎么显示？
把图像增强工作流加入算法？

The author uses ArcFace loss to measure the error of prediction. This loss was proposed for facial recognition in 2018. Other sophisticated approaches have also been published in recent years, such as [ElasticFace](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Boutros_ElasticFace_Elastic_Margin_Loss_for_Deep_Face_Recognition_CVPRW_2022_paper.pdf). author can compare the proposed loss with this approach.

[YOLOV5 模型和代码修改——针对小目标识别](https://blog.csdn.net/weixin_56184890/article/details/119840555)