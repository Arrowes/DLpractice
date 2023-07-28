---
title: DL模型转换及部署：torch > onnx > deploy
date: 2023-06-09 11:36:00
tags:
- 嵌入式
- 深度学习
---

**算法部署**
+ Network selection：
+ Optimization：分组卷积、深度可分离卷积、稀疏卷积
+ Deployment：
<img alt="图 1" src="https://raw.sevencdn.com/Arrowes/Blog/main/images/TDA4VMdeploy.png" width="70%"/>  

Netron神经网络可视化: [软件下载](https://github.com/lutzroeder/netron/releases/tag/v7.0.0), [在线网站](https://netron.app/)



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
<img alt="picture 0" src="https://raw.sevencdn.com/Arrowes/Blog/main/images/DLdeploynetron.png" width="80%"/>  

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







# 量化
量化一般是指把模型的单精度参数（Float32）转化为低精度参数(Int8,Int4)，把推理过程中的浮点运算转化为定点运算。
*（float和int的本质区别在于小数点是否固定）*

浮点数格式 (float32)：$V = (-1)^s×M×2^E$
符号位s|阶码E|尾数M|
---|--|--
1|8|23|

定点数格式 (int8)：
符号位|整数位（设定）|小数位(量化系数)|
---|--|--
1|4|3|

若整数位占4位，小数位占3位，则其最大精度为0.125，最大值为15.875
若整数位占5位，小数位占2位，则其最大精度为0.250，最大值为31.750
$int8=float32∗2^3$
$float32=int8/2^3$


浮点运算在运算过程中，小数点的位置是变动的，而定点运算则是固定不变。如果将浮点数转换成定点数，就可以实现一次读取多个数进行计算（1 float32 = 4 int8），提高了运算效率。

> 8位和16位是指量化的位深度，表示用多少个二进制位来表示每个权重或激活值。在量化时，8位会将每个权重或激活值分成256个不同的离散值，而16位则分为65536个离散值，因此16位的表示范围更广，可以更精确地表示模型中的参数和激活值。但是，使用较高的位深度会增加存储要求和计算成本，因此需要在预测精度和计算开销之间进行权衡。
<img src="https://img2018.cnblogs.com/blog/947235/201905/947235-20190513143437402-715176586.png" width='70%'>

乘一个系数把float类型的小数部分转换成整数部分，然后用这个转换出来的整数进行计算，计算结果再还原成float

<img alt="图 3" src="https://raw.sevencdn.com/Arrowes/Blog/main/images/DLdeployquantized.png" width="80%"/>  

[A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295.pdf)
