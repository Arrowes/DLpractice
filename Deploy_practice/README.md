---
title: DL模型转换及部署：torch > onnx > deploy
date: 2023-06-09 11:36:00
tags:
- 嵌入式
- 深度学习
---
深度学习模型部署相关记录，目前仅开了个头，项目地址：[DLpractice](https://github.com/Arrowes/DLpractice)
<!--more-->

**算法部署**
+ Network selection：
+ Optimization：分组卷积、深度可分离卷积、稀疏卷积
+ Deployment：
<img alt="图 1" src="https://raw.gitmirror.com/Arrowes/Blog/main/images/TDA4VMdeploy.png" width="70%"/>  

Netron神经网络可视化: [软件下载](https://github.com/lutzroeder/netron/releases/tag/v7.0.0), [在线网站](https://netron.app/)

---
# 模型部署
[AI 框架部署方案之模型部署概述](https://zhuanlan.zhihu.com/p/367042545)
[AI 框架部署方案之模型转换](https://zhuanlan.zhihu.com/p/396781295)

学术界负责各种 SOTA(State of the Art) 模型的训练和结构探索，而工业界负责将这些 SOTA 模型应用落地，赋能百业。模型部署一般无需再考虑如何修改训练方式或者修改网络结构以提高模型精度，更多的是需要明确部署的场景、部署方式（中心服务化还是本地终端部署）、模型的优化指标，以及如何提高吞吐率和减少延迟等。

## 模型部署场景
+ 云端部署：模型部署在云端服务器，用户通过网页访问或者 API 接口调用等形式向云端服务器发出请求，云端收到请求后处理并返回结果。 
+ **边缘部署**：主要用于嵌入式设备，主要通过将模型打包封装到 SDK，集成到嵌入式设备，数据的处理和模型推理都在终端设备上执行。

## 模型部署方式
+ Service 部署：主要用于中心服务器云端部署，一般直接以训练的引擎库作为推理服务模式。
+ **SDK 部署**：主要用于嵌入式端部署场景，以 C++ 等语言实现一套高效的前后处理和推理引擎库（高效推理模式下的 Operation/Layer/Module 的实现），用于提供高性能推理能力。此种方式一般需要考虑模型转换（动态图静态化）、模型联合编译等进行深度优化。


|           | SDK部署 | Service部署 |
| --------- | ------- | ------------ |
| 部署环境 | SDK引擎 | 训练框架 |
| 模型语义转换 | 需要进行前后处理和模型的算子重实现 | 一般框架内部负责语义转换 |
| 前后处理对齐算子 | 训练和部署对应两套实现，需要进行算子数值对齐 | 共用算子 |
| 计算优化 | 偏向于挖掘芯片编译器的深度优化能力 | 利用引擎已有训练优化能力 |

<img alt="图 3" src="https://pic2.zhimg.com/80/v2-ffd5625be23ba4fa8a56f5232a3f9c95_720w.webp" width="80%"/>  

## 部署优化指标
成本、功耗、性价比

AI 模型部署到硬件上的成本将极大限制用户的业务承受能力。
成本问题主要聚焦于芯片的选型，比如，对比**寒武纪 MLU220** 和 MLU270，MLU270 主要用作数据中心级的加速卡，其算力和功耗都相对于边缘端的人工智能加速卡MLU220要高。至于 Nvida 推出的 Jetson 和 Tesla T4 也是类似思路，Tesla T4 是主打数据中心的推理加速卡，而 Jetson 则是嵌入式设备的加速卡。对于终端场景，还会根据对算力的需求进一步细分，比如表中给出的高通骁龙芯片，除 GPU 的浮点算力外，还会增加 DSP 以增加定点算力

| 芯片型号          | 算力                                    | 功耗          |
| ---------------- | --------------------------------------- | ------------- |
| Snapdragon 855   | 7 TOPS (DSP) + 954.7 GFLOPs(GPU FP32)   | 10 W          |
| Snapdragon 865   | 15 TOPS (DSP) + 1372.1 GFLOPs(GPU FP32) | 10 W          |
| MLU220           | 8 TOPS (INT8)                           | 8.25 W        |
| MLU270-S4        | 128 TOPS (INT8)                         | 70 W          |
| Jetson-TX2       | 1.30 TOPS (FP16)                        | 7.5 W / 15 W  |
| T4               | 130 TOPS (INT8)                         | 70 W          |

在数据中心服务场景，对于功耗的约束要求相对较低；在边缘终端设备场景，硬件的功耗会影响边缘设备的电池使用时长。因此，对于功耗要求相对较高，一般来说，利用 NPU 等专用优化的加速器单元来处理神经网络等高密度计算，能节省大量功耗。

不同的业务场景对于芯片的选择有所不同，以达到更高的性价比。 从公司业务来看，云端相对更加关注是多路的吞吐量优化需求，而终端场景则更关注单路的延时需要。

在目前主流的 CV 领域，低比特模型相对成熟，且 INT8/INT4 芯片因成本低，且算力比高的原因已被广泛使用；但在NLP或者语音等领域，对于精度的要求较高，低比特模型精度可能会存在难以接受的精度损失，因此 FP16 是相对更优的选择。

在 CV 领域的芯片性价比选型上，在有 INT8/INT4 计算精度的芯片里，主打低精度算力的产品是追求高性价比的主要选择之一，但这也为平衡精度和性价比提出了巨大的挑战。

## 部署流程
模型转换、模型量化压缩、模型打包封装 SDK。

**模型转换**:主要用于模型在不同框架之间的流转，常用于训练和推理场景的连接。目前主流的框架都以 ONNX 或者 caffe 为模型的交换格式；模型转换主要分为计算图生成和计算图转换两大步骤，另外，根据需要，还可以在中间插入计算图优化，对计算机进行推理加速（诸如常见的 CONV/BN 的算子融合），例如去除冗余 op，计算合并等。

+ 计算图生成：通过一次 inference 并追踪记录的方式，将用户的模型完整地翻译成静态的表达。在模型 inference 的过程中，框架会记录执行算子的类型、输入输出、超参、参数和调用该算子的模型层次，最后把 inference 过程中得到的算子信息和模型信息结合得到最终的静态计算图。
目前使用广泛的训练框架 PyTorch 使用的都是动态图，这是由于动态图的表达形式更易于用户快速实现并迭代算法。 动态图框架会逐条解释，逐条执行模型代码来运行模型，而计算图生成的本质是**把动态图模型静态表达出来**。 PyTorch 的torchscript、ONNX、fx 模块都是基于模型静态表达来开发的。目前常见的建立模型静态表达的方法有以下三种：
    + 代码语义分析：通过分析用户代码来解析模型结构，建立模型静态表达。
    + 模型对象分析：通过模型对象中包含的成员变量，来确定模型算子组成，建立模型静态表达。
    + **模型运行追踪**：运行模型并记录过程中的算子信息、数据流动，建立模型静态表达。
    上面这三种方法在适用范围、静态抽象能力等方面各有优劣。目前训练框架都主要使用模型运行追踪的方式来生成计算图：在模型inference 的过程中，框架会记录执行算子的类型、输入输出、超参、参数等算子信息，最后把 inference 过程中得到的算子节点信息和模型信息结合得到最终的静态计算图。
    然而，很多时候，用户的一段代码可能涉及非框架底层的计算，涉及外部库的计算，训练框架自身是无法追踪记录到的。这个时候我们可以把这部分代码作为一个**自定义算子**，由用户定义这个算子在计算图中作为一个节点所记录的信息。实际实现时，这些计算会被写到一个 Function 或者 Module 中，然后用户在 Function 或者 Module 中定义这个计算对应的计算节点的信息表达，这样每次调用这个定义好的 Function 或者 Module，就能对应在计算图中记录相应的算子信息。此外，还有很多其他场景会产生这种需要，例如你的几个计算组成了一个常见的函数，可以有更高层的表达，这个时候也可以使用自定义算子来简化计算图的表达。

+ 计算图转换：计算图转换到目标格式就是去解析静态计算图，根据计算图的定义和目标格式的定义，去做转换和对齐。这里的主要的工作就是通用的优化和转换，以及大量特殊情况的处理。
    + **计算图转换到 ONNX**：ONNX 官方定义了算子集 opset，并且随着 ONNX 的演进，opset 版本的迭代伴随着算子支持列表和算子表达形式的改动，因此针对不同的 opset 也需要有多后端 ONNX 的支持。另一方面，对于在 opset 之外的算子，用户需要自己注册定义算子在 ONNX 的表达信息（输入、输出、超参等）。
    另一方面，推理框架对于 ONNX 官方 opset 往往也不是完全支持，会有自己的一些取舍。所以对于 ONNX 模型，往往需要用相关的 simplifier 进行模型预处理优化，围绕这一方面模型转换或者部署框架的工程侧也有不少的相关工作。

和五花八门的芯片等端侧硬件相比，x86 和 CUDA 平台是普及率最高的平台，因此如果是出于部署测试、转换精度确认、量化等需要，一个能够在 x86 或者 CUDA 平台运行的 runtime 是非常必要的。对此，支持 ONNX 格式的部署框架一般会基于 onnxruntime（微软出品的一个具有 ONNX 执行能力的框架）进行扩展，支持 caffe 格式的部署框架一般会基于原生 caffe 进行扩展。通过 onnxruntime 和 caffe 的推理运行能力，来提供在 x86 或者 CUDA 平台上和硬件平台相同算子表达层次的运行能力。当然还有一些生态较好的部署框架，他们自己提供算子表达能力和计算精度与硬件一致的 x86 或 CUDA 平台的模拟器。


**模型量化压缩**：终端场景中，一般会有内存和速度的考虑，因此会要求模型尽量小，同时保证较高的吞吐率。除了人工针对嵌入式设备设计合适的模型，如 MobileNet 系列，通过 NAS(Neural Architecture Search) 自动搜索小模型，以及通过蒸馏/剪枝的方式压缩模型外，一般还会使用量化来达到减小模型规模和加速的目的。

量化的过程主要是将原始浮点 FP32 训练出来的模型压缩到定点 INT8(或者 INT4/INT1) 的模型，由于 INT8 只需要 8 比特来表示，因此相对于 32 比特的浮点，其模型规模理论上可以直接降为原来的 1/4，这种压缩率是非常直观的。
另外，大部分终端设备都会有专用的定点计算单元，通过低比特指令实现的低精度算子，速度上会有很大的提升，当然，这部分还依赖协同体系结构和算法来获得更大的加速。
+ 量化训练（QAT, Quantization Aware Training）：通过对模型插入伪量化算子（这些算子用来模拟低精度运算的逻辑），通过梯度下降等优化方式在原始浮点模型上进行微调，从来调整参数得到精度符合预期的模型。量化训练基于原始浮点模型的训练逻辑进行训练，理论上更能保证收敛到原始模型的精度，但需要精细调参且生产周期较长；
+ 离线量化：主要是通过少量校准数据集（从原始数据集中挑选 100-1000 张图，不需要训练样本的标签）获得网络的 activation 分布，通过统计手段或者优化浮点和定点输出的分布来获得量化参数，从而获取最终部署的模型。离线量化只需要基于少量校准数据，因此生产周期短且更加灵活，缺点是精度可能略逊于量化训练。
 实际落地过程中，发现大部分模型通过离线量化就可以获得不错的模型精度（1% 以内的精度损失，当然这部分精度的提升也得益于优化策略的加持），剩下少部分模型可能需要通过量化训练来弥补精度损失，因此实际业务中会结合两者的优劣来应用。

两大难点：一是如何平衡模型的吞吐率和精度，二是如何结合推理引擎充分挖掘芯片的能力。 比特数越低其吞吐率可能会越大，但其精度损失可能也会越大，因此，如何通过算法提升精度至关重要，这也是组内的主要工作之一。另外，压缩到低比特，某些情况下吞吐率未必会提升，还需要结合推理引擎优化一起对模型进行图优化，甚至有时候会反馈如何进行网络设计，因此会是一个算法与工程迭代的过程。

**模型打包封装 SDK**：实际业务落地过程中，模型可能只是产品流程中的一环，用于实现某些特定功能，其输出可能会用于流程的下一环。因此，模型打包会将模型的前后处理，一个或者多个模型整合到一起，再加入描述性的文件（前后处理的参数、模型相关参数、模型格式和版本等）来实现一个完整的功能。因此，SDK 除了需要一些通用前后处理的高效实现，对齐训练时的前后处理逻辑，还需要具有足够好的扩展性来应对不同的场景，方便业务线扩展新的功能。可以看到，模型打包过程更多是模型的进一步组装，将不同模型组装在一起，当需要使用的时候将这些内容解析成整个流程（pipeline）的不同阶段（stage），从而实现整个产品功能。

另外，考虑到模型很大程度是研究员的研究成果，对外涉及保密问题，因此会对模型进行加密，以保证其安全性。加密算法的选择需要根据实际业务需求来决定，诸如不同加密算法其加解密效率不一样，加解密是否有中心验证服务器，其核心都是为了保护研究成果。


---

# 量化
量化一般是指把模型的单精度参数（Float32）转化为低精度参数(Int8,Int4)，把推理过程中的浮点运算转化为定点运算。
*（float和int的本质区别在于小数点是否固定）*

浮点数格式 (float32)：$$V = (-1)^s×M×2^E$$
符号位s|阶码E|尾数M|
---|--|--
1|8|23|

定点数格式 (int8)：
符号位|整数位（设定）|小数位(量化系数)|
---|--|--
1|4|3|

若整数位占4位，小数位占3位，则其最大精度为0.125，最大值为15.875
若整数位占5位，小数位占2位，则其最大精度为0.250，最大值为31.750
$$int8=float32∗2^3$$
$$float32=int8/2^3$$


浮点运算在运算过程中，小数点的位置是变动的，而定点运算则是固定不变。如果将浮点数转换成定点数，就可以实现一次读取多个数进行计算（1 float32 = 4 int8），提高了运算效率。

> 8位和16位是指量化的位深度，表示用多少个二进制位来表示每个权重或激活值。在量化时，8位会将每个权重或激活值分成256个不同的离散值，而16位则分为65536个离散值，因此16位的表示范围更广，可以更精确地表示模型中的参数和激活值。但是，使用较高的位深度会增加存储要求和计算成本，因此需要在预测精度和计算开销之间进行权衡。

<img alt="picture 1" src="https://raw.gitmirror.com/Arrowes/Blog/main/images/DLdeploy-16to8bit.png" width="70%"/>  



乘一个系数把float类型的小数部分转换成整数部分，然后用这个转换出来的整数进行计算，计算结果再还原成float

<img alt="图 3" src="https://raw.gitmirror.com/Arrowes/Blog/main/images/DLdeployquantized.png" width="80%"/>  

[A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295.pdf)

[AI 框架部署方案之模型量化概述](https://zhuanlan.zhihu.com/p/354921065)
[AI 框架部署方案之模型量化的损失分析](https://zhuanlan.zhihu.com/p/400927037)
___

# 模型部署的软件设计（以商汤的MMdeploy部署工具箱为例）
## 模型转换器设计
[千行百业智能化落地，MMDeploy 助你一"部"到位 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/450342651)
<img alt="图 3" src="https://pic1.zhimg.com/80/v2-a076d9317d2167d9d6d8898e0db0fd7c_1440w.webp" width="80%"/>  
模型转换器的具体步骤为：

1. 把 PyTorch 转换成 ONNX 模型；
2. 对 ONNX 模型进行优化；
3. 把 ONNX 模型转换成后端推理引擎支持的模型格式;
4. （可选）把模型转换中的 meta 信息和后端模型打包成 SDK 模型。

在传统部署流水线中，兼容性是最难以解决的瓶颈。针对这些问题，MMDeploy 在模型转换器中添加了模块重写、模型分块和自定义算子这三大功能
+ 模块重写——有效代码替换
针对部分 Python 代码无法直接转换成 ONNX 的问题，MMDeploy 使用重写机制实现了函数、模块、符号表等三种粒度的代码替换，有效地适配 ONNX。
+ 模型分块——精准切除冗余
针对部分模型的逻辑过于复杂，在后端里无法支持的问题，MMDeploy 使用了模型分块机制，能像手术刀一样精准切除掉模型中难以转换的部分，把原模型分成多个子模型，分别转换。这些被去掉的逻辑会在 SDK 中实现。
+ 自定义算子——扩展引擎能力
OpenMMLab 实现了一些新算子，这些算子在 ONNX 或者后端中没有支持。针对这个问题，MMDeploy 把自定义算子在多个后端上进行了实现，扩充了推理引擎的表达能力。

## 应用开发工具包 SDK
<img alt="图 3" src="https://pic2.zhimg.com/80/v2-5618bc32c6018dbe6b7419555c373445_1440w.webp" width="80%"/>  

+ 接口层
SDK 为每种视觉任务均提供一组 C API。目前开放了分类、检测、分割、超分、文字检测、文字识别等几类任务的接口。 SDK 充分考虑了接口的易用性和友好性。每组接口均只由 ‘创建句柄’、‘应用句柄’、‘销毁数据’ 和 ‘销毁句柄’ 等函数组成。用法简单、便于集成。
+ 流水线层
SDK 把模型推理统一抽象为计算流水线，包括前处理、网络推理和后处理。对流水线的描述在 SDK Model 的 meta 信息中。使用 Model Converter 转换模型时，加入 --dump-info 命令，即可自动生成。 不仅是单模型，SDK同样可把流水线拓展到多模型推理场景。比如在检测任务后，接入识别任务。
+ 组件层
组件层为流水线中的节点提供具体的功能。SDK 定义了3类组件，
    + 设备组件（Device）：对硬件设备以及 runtime 的抽象
    + 模型组件（Model）：支持 SDK Model 不同的文件格式
    + 任务组件（Task）：模型推理过程中，流水线的最小执行单元。它包括:
        + 预处理（preprocess）：与 OpenMMLab Transform 算子对齐，比如 Resize、Crop、Pad、Normalize等等。每种算子均提供了 cpu、cuda 两种实现方式。
        + 网络推理引擎（net）：对推理引擎的封装。目前，SDK 可以接入5种推理引擎：PPL.NN, TensorRT, ONNX Runtime, NCNN 和 OpenVINO
        + 后处理（postprocess）：对应与 OpenMMLab 各算法库的后处理功能。
+ 核心层
核心层是 SDK 的基石，定义了 SDK 最基础、最核心的数据结构。

# ONNX
Open Neural Network Exchange 开源机器学习通用中间格式，兼容各种深度学习框架、推理引擎、终端硬件、操作系统，是深度学习框架到推理引擎的桥梁 
链接：[ONNX](https://onnx.ai)，[Github](https://github.com/onnx/onnx)，[ONNX Runtime](https://onnxruntime.ai/)，[ONNX Runtime Web](https://onnx.coderai.cn)

[TORCH.ONNX](https://pytorch.org/docs/stable/onnx.html)，[Github](https://github.com/pytorch/pytorch/tree/main/torch/onnx)
Pytorch 模型导出使用自带的接口：`torch.onnx.export`
 PyTorch 转 ONNX，实际上就是把每个 PyTorch 的操作**映射**成了 ONNX 定义的**算子**。PyTorch 对 ONNX 的算子支持:[官方算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)


在转换普通的torch.nn.Module模型时，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子。要使 PyTorch 算子顺利转换到 ONNX ，我们需要保证：
> 1.算子在 PyTorch 中有实现
2.有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
3.ONNX 有相应的算子

## 以超分辨率模型为例
参考：[模型部署那些事](https://www.zhihu.com/column/c_1497987564452114432)
以超分辨率模型为例，实现pytorch模型转onnx
其中， PyTorch 的 interpolate 插值算子可以在运行阶段选择放大倍数，但该算子不兼容，需要**自定义算子**:
```py
class NewInterpolate(torch.autograd.Function):
    # 自定义的插值算子，继承自torch.autograd.Function
    @staticmethod
    def symbolic(g, input, scales):
        # 静态方法，用于定义符号图的构建过程, g: 符号图构建器, input: 输入张量, scales: 缩放因子
        #ONNX 算子的具体定义由 g.op 实现。g.op 的每个参数都可以映射到 ONNX 中的算子属性
        #对于其他参数，可以照着 Resize 算子文档填
        return g.op("Resize",  # 使用Resize操作
                    input,  # 输入张量
                    g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),  # 空的常量张量
                    scales,  # 缩放因子
                    coordinate_transformation_mode_s="pytorch_half_pixel",  # 坐标转换模式为pytorch_half_pixel
                    cubic_coeff_a_f=-0.75,  # cubic插值的系数a为-0.75
                    mode_s='cubic',  # 插值模式为cubic
                    nearest_mode_s="floor")  # 最近邻插值模式为floor

    @staticmethod
    def forward(ctx, input, scales):    #算子的推理行为由算子的 foward 方法决定
        scales = scales.tolist()[-2:]   #截取输入张量的后两个元素,把 [1, 1, w, h] 格式的输入对接到原来的 interpolate 函数上
        return interpolate(input,   #把这两个元素以 list 的格式传入 interpolate 的 scale_factor 参数。
                           scale_factor=scales,
                           mode='bicubic',
                           align_corners=False)
```

<details>
    <summary>SRCNN超分辨率代码</summary>

```py
class StrangeSuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):
        x = NewInterpolate.apply(x, upscale_factor)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


def init_torch_model():
    torch_model = StrangeSuperResolutionNet()

    state_dict = torch.load('srcnn.pth')['state_dict']

    # Adapt the checkpoint
    for old_key in list(state_dict.keys()):
        new_key = '.'.join(old_key.split('.')[1:])
        state_dict[new_key] = state_dict.pop(old_key)

    torch_model.load_state_dict(state_dict)
    torch_model.eval()
    return torch_model


model = init_torch_model()
factor = torch.tensor([1, 1, 3, 3], dtype=torch.float)

input_img = cv2.imread('face.png').astype(np.float32)

# HWC to NCHW
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)

# Inference
torch_output = model(torch.from_numpy(input_img), factor).detach().numpy()

# NCHW to HWC
torch_output = np.squeeze(torch_output, 0)
torch_output = np.clip(torch_output, 0, 255)
torch_output = np.transpose(torch_output, [1, 2, 0]).astype(np.uint8)

# Show image
cv2.imwrite("face_torch2.png", torch_output)
input_img1 = cv2.imread('face.png')
cv2.imshow("Input Image", input_img1)
cv2.imshow("Torch Output", torch_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
</details>

---
模型转换为ONNX，验证正确性，运行推理：
```py
# pth2onnx
x = torch.randn(1, 3, 256, 256)
# 一种叫做追踪（trace）的模型转换方法：给定一组输入，再实际执行一遍模型，即把这组输入对应的计算图记录下来，保存为 ONNX 格式
with torch.no_grad():
    torch.onnx.export(model, (x, factor),
                      "srcnn2.onnx",
                      opset_version=11,
                      input_names=['input', 'factor'],
                      output_names=['output'])


# 验证onnx, 此外可以使用Netron可视化检查网络结构
onnx_model = onnx.load("srcnn.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")


# 选择放大倍数，运行ONNX Runtime 推理
input_factor = np.array([1, 1, 5, 5], dtype=np.float32)
ort_session = onnxruntime.InferenceSession("srcnn2.onnx")   # 用于获取一个 ONNX Runtime 推理器
ort_inputs = {'input': input_img, 'factor': input_factor}
ort_output = ort_session.run(None, ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite("face_torch2_run.png", ort_output)  # 生成上采样图片，运行成功
```
<img alt="picture 0" src="https://raw.gitmirror.com/Arrowes/Blog/main/images/DLdeploynetron.png" width="80%"/>  

## torch.onnx.export模型转换接口
[torch.onnx ‒ PyTorch 1.11.0 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/onnx.html%23functions)
[TorchScript](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/jit.html) 是一种序列化和优化 PyTorch 模型的格式，在优化过程中，一个`torch.nn.Module`模型会被转换成 TorchScript 的 `torch.jit.ScriptModule`模型。
而要把普通 PyTorch 模型转一个 TorchScript 模型，有跟踪（trace）和记录（script）两种导出计算图的方法：
+ trace: 以上一节为例，跟踪法只能通过实际运行一遍模型的方法导出模型的静态图，即无法识别出模型中的控制流（如循环）,对于循环中不同的n, ONNX 模型的结构是不一样的
+ script: 记录法则能通过解析模型来正确记录所有的控制流,模型不需要实际运行，用 Loop 节点来表示循环

```py
def export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL, 
           input_names=None, output_names=None, aten=False, export_raw_ir=False, 
           operator_export_type=None, opset_version=None, _retain_param_name=True, 
           do_constant_folding=True, example_outputs=None, strip_doc_string=True, 
           dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, 
           enable_onnx_checker=True, use_external_data_format=False): 

# model: 模型， args：输入， f：导出文件名，
# export_params：是否存储模型权重， ONNX 是用同一个文件表示记录模型的结构和权重的。
# input_names, output_names：设置输入和输出张量的名称。如果不设置的话，会自动分配一些简单的名字（如数字）
# opset_version：转换时参考哪个 ONNX 算子集版本，默认为 9。
# dynamic_axes：指定输入输出张量的哪些维度是动态的。为了效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变），必要时需要显式地指明输入输出张量的哪几个维度的大小是可变的。
```

## 自定义算子
-   PyTorch 算子
    -   组合现有算子
    -   添加 TorchScript 算子
    -   添加普通 C++ 拓展算子
-   映射方法
    -   为 ATen 算子添加符号函数
    -   为 TorchScript 算子添加符号函数
    -   封装成 `torch.autograd.Function` 并添加符号函数
-   ONNX 算子
    -   使用现有 ONNX 算子
    -   定义新 ONNX 算子

[模型部署入门教程（四）：在 PyTorch 中支持更多 ONNX 算子](https://zhuanlan.zhihu.com/p/513387413)
