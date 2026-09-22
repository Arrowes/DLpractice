---
title: TDA4④：部署自定义深度学习模型
date: 2023-07-07 10:40:00
tags:
- 嵌入式
- 深度学习
---
自定义深度学习模型的转换、编译及部署流程，使用了三种不同的编译工具：TIDL Importer，Edge AI Studio，EdgeAI-TIDL-Tools
<!--more-->

接上一篇：[TDA4③：YOLOX的模型转换与SK板端运行](https://wangyujie.space/TDA4VM3/)

TI文档中对yolo、mobilenet、resnet等主流深度学习模型支持十分完善，相关开箱即用的文件在 [Modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo) 中，但有关自定义模型的编译和部署内容很少，只能利用例程和提供的工具进行尝试。

深度学习模型基于TI板端运行要有几个组件：
1.  **model**：这个目录包含了要进行推理的模型（.onnx, *.prototxt）
2.  **artifacts**：这个目录包含了模型编译后生成的文件。这些文件可以用Edge AI TIDL Tools来生成和验证
3.  **param.yaml**：配置文件，提供了模型的基本信息，以及相关的预处理和后处理参数
4.  \***dataset.yaml**：配置文件，说明了用于模型训练的数据集的细节
5.  \***run.log**：这是模型的运行日志

[edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark): Custom model benchmark can also be easily done (please refer to the documentation and example). 
Uses [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) for model compilation and inference

# 网络结构的修改与适配
edgeai-tidl-tools与edge ai studio的编译结果可以结合onnx模型在arm上运行，因此可以有不支持的网络层（有性能损失），但若使用TIDL Importer编译，则只能转换完全支持TIDL的网络结构，因此前期将网络中不支持的层替换是最好的，

此处以YOLOX的Backbone为例，修改不支持的层：slice, ~~Resize_206, Resize_229~~(resize在version13不支持，11支持), MaxPool(在11只支持kernel sizes: 3x3,2x2,1x1)

TIDL支持的算子见：[supported_ops_rts_versions](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/supported_ops_rts_versions.md)
ONNX算子版本见：[onnx/docs/Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

| TIDL Layer Type| ONNX Ops| TFLite Ops| Notes |
|:--------|:--------|:----------|:------|
TIDL_SliceLayer	|Split|	NA	|Only channel wise slice is supported
TIDL_ResizeLayer	|UpSample|	RESIZE_NEAREST_NEIGHBOR|RESIZE_BILINEAR	Only power of 2 and symmetric resize is supported
TIDL_PoolingLayer	|MaxPool, AveragePool, GlobalAveragePool|MAX_POOL_2D, AVERAGE_POOL_2D, MEAN	|Pooling has been validated for the following kernel sizes: 3x3,2x2,1x1, with a maximum stride of 2

修改网络中三处不支持的层以支持TIDL：
```py
(1,1,256,128) --> Slice + Concat --> (1,4,128,64)
#Slice+Concat参照TI_YOLOX, 替换为Conv + Relu

(1,64,8,4)  --> Resize_206 --> (1,64,16,8)
(1,32,16,8) --> Resize_229 --> (1,32,32,16)
#resize理论上支持，此处原因待排查
#原因是onnx转换时opset=13，应为opset=11，网络无需修改

#opset vertion改为11后 MaxPool 需要拆分为 kernel=3的组合
maxpool(k=5, s=1) -> replaced with two maxpool(k=3,s=1)
maxpool(k=9, s=1) -> replaced with four maxpool(k=3,s=1)
maxpool(k=13, s=1)-> replaced with six maxpool(k=3,s=1)
```

参考TI官方对YOLOx的更改 [edgeai-yolox/README_2d_od](https://github.com/TexasInstruments/edgeai-yolox/blob/main/README_2d_od.md)，将Slice替换为一个卷积层，再对MaxPool拆分，最后激活函数Silu替换为Relu，再重新训练，得到新模型，设为opset_version=11重新导出onnx编译后，即可只生成2个bin文件（net+io），完全的支持tidl运行加速；

# ONNX模型转换及推理
使用`torch.onnx.export(model, input, "XXX.onnx", verbose=False, export_params=True, opset_version=11)`得到 `.onnx`；
> 注意要确保加载的模型是一个完整的PyTorch模型对象，而不是一个包含模型权重的字典, 否则会报错`'dict' object has no attribute 'modules'`；
因此需要在项目保存`.pth`模型文件时设置同时*保存网络结构*，或者在项目代码中*导入完整模型*后使用`torch.onnx.export`
**opset_version只支持到13**，导出默认是14，会报错
opset_version为13时不支持resize, 现改为**11**

使用ONNX Runtime 运行推理，验证模型转换的正确性
```py
import numpy as np    
import onnxruntime    
from PIL import Image
import onnx
import cv2
import matplotlib.pyplot as plt
import torch

#导入模型和推理图片
model_path = "./XXX.onnx"
input_file="1.jpg"
session = onnxruntime.InferenceSession(model_path, None)

# get the name of the first input of the model
input_name = session.get_inputs()[0].name  
input_details  = session.get_inputs()
print("Model input details:")
for i in input_details:
    print(i)
output_details = session.get_outputs()
print("Model output details:", )
for i in output_details:
    print(i)

input_shape = input_details[0].shape
input_height, input_width = input_shape[2:]

# Pre-Process input
img_bgr = cv2.imread(input_file)
print("image size:", img_bgr.shape)
img_bgr2 = cv2.resize(img_bgr, ( input_width,input_height))
print("image resize:", img_bgr2.shape)
img_rgb = img_bgr2[:,:,::-1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 预处理-归一化
input_tensor = img_rgb / 255    # 预处理-构造输入 Tensor
input_tensor = np.expand_dims(input_tensor, axis=0) # 加 batch 维度
input_tensor = input_tensor.transpose((0, 3, 1, 2)) # N, C, H, W
input_tensor = np.ascontiguousarray(input_tensor)   # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
input_tensor = torch.from_numpy(input_tensor).to(device).float() # 转 Pytorch Tensor
input_tensor = input_tensor[:, :1, :, :]    #[1, "1", 384, 128]
print(input_tensor.shape)

#Run inference session
raw_result = session.run([], {input_name: input_tensor.numpy()})
for result in raw_result:
    print("result shape:", result.shape)
```
`print(result)` :正常应该输出正确的推理结果，如果数值全都一样(-4.59512)，可能是没有检测到有效的目标或者模型效果太差

# TIDL 编译转换
得到onnx相关文件后，使用ti提供的工具进行编译和推理，这里采用三种不同的模型转换方法： 
+ [TIDL Importer](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/06_01_01_12/exports/docs/tidl_j7_01_00_01_00/ti_dl/docs/user_guide_html/md_tidl_model_import.html) ：部署于EVM板，网络结构需要全部支持TIDL
+ [Edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools/tree/08_06_00_05)：可部署于SK板，需要onnx运行环境，配置灵活，网络结构要求少
+ [Edge AI Studio](https://dev.ti.com/edgeaistudio/)：TIDL tools的在线版本

## TIDL Importer
TIDL Importer 是RTOS SDK中提供的导入工具，需要网络结构完全支持tidl，以使模型都通过tidl加速（即转换只生成net,io 2个bin文件）

下面的流程重构了文件夹架构，原文件跳来跳去改起来很麻烦，就合并到了XXX文件夹，原文件路径可参考上一篇官方例程： [TDA4③_使用TIDL Importer导入YOLOX](https://wangyujie.space/TDA4VM3/#a-%E4%BD%BF%E7%94%A8TIDL-Importer-by-RTOS-SDK)

1. 配置文件：新建文件夹：`SDK/${TIDL_INSTALL_PATH}/ti_dl/test/testvecs/XXX`
    拷贝 onnx 文件至 XXX 文件夹 (*此处是自定义模型，不使用prototxt, 经测试可以正常编译*)
    ```sh
    #XXX文件夹结构
    ├── detection_list.txt
    ├── device_configs
    │   ├── am62a_config.cfg
    │   ├── j721e_config.cfg
    │   ├── j721s2_config.cfg
    │   └── j784s4_config.cfg
    ├── output
    ├── indata
    │   └── 1.jpg
    ├── XXX.onnx
    └── tidl_import_XXX.txt
    ```

2. 编写转换配置文件：新建**tidl_import_XXX.txt**，可参考同目录下其他例程，详细参数配置见[文档](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/06_01_01_12/exports/docs/tidl_j7_01_00_01_00/ti_dl/docs/user_guide_html/md_tidl_model_import.html)
```sh
#tidl_import_XXX.txt:
modelType          = 2
numParamBits       = 8
numFeatureBits     = 8
quantizationStyle  = 3
inputNetFile       = "../../test/testvecs/XXX/XXX_yolox_221_sig_11.onnx"
outputNetFile      = "../../test/testvecs/XXX/output/825_tidl_net_sig_SDK8_6.bin"
outputParamsFile   = "../../test/testvecs/XXX/output/825_tidl_io__sig_SDK8_6"
inDataNorm  = 1
inMean = 0 0 0
inScale = 0.003921568627 0.003921568627 0.003921568627
inDataFormat = 1
inWidth  = 128
inHeight = 256 
inNumChannels = 3
numFrames = 50
inData  =   "../../test/testvecs/XXX/detection_list.txt"
perfSimConfig = ../../test/testvecs/XXX/device_configs/j721s2_config.cfg
debugTraceLevel = 1
```
>Debug:
`inData`配置数据输入(回车分隔)，数量与`numFrames`要匹配；
`perfSimConfig`选择对应设备的配置文件；
`inScale`配置太大可能导致tensor不匹配
`metaLayersNamesList`注释掉, 除非与TI提供的元架构相同；

3. 执行编译，得到可执行文件 `.bin`
    ```sh
    export TIDL_INSTALL_PATH=/home/wyj/sda2/TAD4VL_SKD_8_5/ti-processor-sdk-rtos-j721s2-evm-08_05_00_11/tidl_j721s2_08_05_00_16
    cd ${TIDL_INSTALL_PATH}/ti_dl/utils/tidlModelImport
    ./out/tidl_model_import.out ${TIDL_INSTALL_PATH}/ti_dl/test/testvecs/XXX/tidl_import_yolox.txt
    #successful Memory allocation
    #../../test/testvecs/XXX/output/生成的文件分析：
    tidl_net_XXX.bin        #Compiled network file 网络模型数据
    tidl_io_XXX.bin         #Compiled I/O file 网络输入配置文件
    tidl_net_XXX.bin.svg    #tidlModelGraphviz tool生成的网络图
    tidl_out.png, tidl_out.txt  #执行的目标检测测试结果
    ```

4. TIDL运行(inference)
[TI Deep Learning Library User Guide: TIDL Inference](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/06_01_01_12/exports/docs/tidl_j7_01_00_01_00/ti_dl/docs/user_guide_html/md_tidl_sample_test.html)
    ```sh
    #在文件ti_dl/test/testvecs/config/config_list.txt顶部加入:
    1 testvecs/XXX/tidl_infer_XXX.txt
    0

    #新建tidl_infer_yolox.txt:
    inFileFormat    = 2
    numFrames   = 10
    netBinFile      = "testvecs/XXX/output/825_tidl_net_sig_SDK8_6.bin"
    ioConfigFile   = "testvecs/XXX/output/825_tidl_io_sig_SDK8_61.bin"
    inData  =   testvecs/XXX/detection_list.txt
    outData =   testvecs/XXX/infer_out/inference.bin
    inResizeMode = 0
    #0 : Disable, 1- Classification top 1 and 5 accuracy, 2 – Draw bounding box for OD, 3 - Pixel level color blending
    postProcType = 2
    debugTraceLevel = 1
    writeTraceLevel = 0
    writeOutput = 1

    #运行，结果在ti_dl/test/testvecs/output/
    cd ${TIDL_INSTALL_PATH}/ti_dl/test && ./PC_dsp_test_dl_algo.out
    ```

## Edge AI Studio
参考yolox的编译过程：[YOLOX的模型转换与SK板端运行](https://wangyujie.space/TDA4VM3/#b-%E4%BD%BF%E7%94%A8TIDL-Tools%EF%BC%88by-Edge-AI-Studio%EF%BC%89)，修改数据预处理与compile_options部分，最后重写画框部分（optional）

> **Debug:**
`[ONNXRuntimeError] : 6 ... `: compile_options中设置deny_list，剔除不支持的层，如`'Slice'`，TIDL支持的算子见：[supported_ops_rts_versions](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/supported_ops_rts_versions.md)    (resize支持2*操作)
compile_options中要注释掉object_detection的配置

打包下载编译生成的工件：
```py
#Pack.ipynb
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

## [EdgeAI-TIDL-Tools](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/08_06_00_05/docs/custom_model_evaluation.md)
环境搭建见：[TDA4②](https://wangyujie.space/TDA4VM2/#EdgeAI-TIDL-Tools)

研读 [edgeai-tidl-tools/examples/osrt_python/ort/onnxrt_ep.py](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/08_06_00_05/examples/osrt_python/ort/onnxrt_ep.py):
进入搭建好的环境：（例）`pyenv activate benchmark` 或 `conda activate tidl`
运行：`./scripts/run_python_examples.sh`
下面基于例程进行基本的修改以编译运行自定义模型, 至少需要修改四个文件：
```sh
#新建运行脚本./script/run.sh
CURDIR=`pwd`
export SOC=am68pa
export TIDL_TOOLS_PATH=$(pwd)/tidl_tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH
export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
    cd $CURDIR/examples/osrt_python/ort
    #python3 onnxrt_ep.py -c
    python3 onnxrt_ep.py
    #python3 onnxrt_ep.py -d

#修改examples/osrt_python/ort/onnxrt_ep.py
def infer_image(sess, image_files, config): #此处修改模型输入数据格式
models = ['custom_model_name']  #修改对应的模型名称

#修改examples/osrt_python/model_configs.py 导入并配置模型
#onnx文件移入model/public文件夹
    'custom_model_name' : {
        'model_path' : os.path.join(models_base_path, 'custom_model_name.onnx'),
        'source' : {'model_url': 'https..XXX./.onnx', 'opt': True,  'infer_shape' : True},
        'mean': [123.675, 116.28, 103.53],
        'scale' : [0.017125, 0.017507, 0.017429],
        'num_images' : numImages,
        'num_classes': 1000,
        'session_name' : 'onnxrt' ,
        'model_type': 'classification'
    },

#examples/osrt_python/common_utils.py 配置编译选项
tensor_bits = 8
debug_level = 0
max_num_subgraphs =16   #16
accuracy_level = 1
calibration_frames = 3  #3
calibration_iterations = 5  #10
output_feature_16bit_names_list = ""#"conv1_2, fire9/concat_1"
params_16bit_names_list = "" #"fire3/squeeze1x1_2"
mixed_precision_factor = -1
quantization_scale_type = 0
high_resolution_optimization = 0
pre_batchnorm_fold = 1
ti_internal_nc_flag = 1601

"deny_list":"Slice", #"MaxPool"

#运行编译
./scripts/run.sh
```
配置编译选项文档：[edgeai-tidl-tools/examples/osrt_python/README.md](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md#optional-options)

> **Debug:**
有些模型可能要到model_configs中找到链接手动下载放入models/public
`'TIDLCompilationProvider' is not in available:`环境问题，没有进入配置好的环境，正常应该是: `Available execution providers :  ['TIDLExecutionProvider', 'TIDLCompilationProvider', 'CPUExecutionProvider']`

### onnxrt_ep.py详解
[edgeai-tidl-tools/examples/osrt_python/ort/onnxrt_ep.py](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/08_06_00_05/examples/osrt_python/ort/onnxrt_ep.py) 是主要运行文件，也是修改的最多的部分，因此梳理此处代码有助于理解*tidl编译和运行的全流程*。
> **Debug**:
其中容易出问题的是预处理部分，image size不对很容易出问题。
替换的某些测试图片读不进去导致报错，~~原理未知~~ 权限问题，sudo nautilus 右键属性更改读写权限

<details>
<summary>onnxrt_ep.py code</summary>

```py
import onnxruntime as rt
import time
import os
import sys
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse
import re
import multiprocessing
import platform

import cv2
import torchvision
from postprogress import *

# directory reach, 获取当前目录和父目录
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)# setting path，将父级目录路径添加到系统路径中，以供后续导入模块使用
from common_utils import *
from model_configs import *
from postprogress import *

# 编译基本选项
required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}

parser = argparse.ArgumentParser()  # 实例化一个参数解析器
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
parser.add_argument('-z','--run_model_zoo', action='store_true',  help='Run model zoo models')
args = parser.parse_args()  # 解析命令行参数
os.environ["TIDL_RT_PERFSTATS"] = "1"   # 设置环境变量 TIDL_RT_PERFSTATS 的值为 "1"

so = rt.SessionOptions()    # 创建一个会话选项对象

print("Available execution providers : ", rt.get_available_providers()) #可用的执行单元
#编译用图片
calib_images = ['../../../test_data/line_test_images.jpg',
'../../../test_data/line_test_images2.jpg',
'../../../test_data/line_test_images3.jpg'
]
#测试用图片
test_images =  ['../../../test_data/line_test_images.jpg',
'../../../test_data/line_test_images2.jpg',
'../../../test_data/line_test_images3.jpg'] 

sem = multiprocessing.Semaphore(0)  # 创建
if platform.machine() == 'aarch64': #检查是否在板端
    ncpus = 1
else:
    ncpus = os.cpu_count()
idx = 0
nthreads = 0
run_count = 0

if "SOC" in os.environ: #检查是否设置了SOC环境变量，无则exit
    SOC = os.environ["SOC"]
else:
    print("Please export SOC var to proceed")
    exit(-1)
if (platform.machine() == 'aarch64'  and args.compile == True): #若在板端且需要编译，exit
    print("Compilation of models is only supported on x86 machine \n\
        Please do the compilation on PC and copy artifacts for running on TIDL devices " )
    exit(-1)
if(SOC == "am62"):
    args.disable_offload = True
    args.compile = False

#计算benchmark
def get_benchmark_output(interpreter):
    benchmark_dict = interpreter.get_TI_benchmark_data()    # 获取模型推理的统计数据字典
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if 'proc_start' in stat:
            value = stat.split("ts:subgraph_")
            value = value[1].split("_proc_start")
            subgraphIds.append(value[0])
    for i in range(len(subgraphIds)):        # 计算处理时间、拷贝输入时间和拷贝输出时间
        proc_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
        copy_time += cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    totaltime = benchmark_dict['ts:run_end'] -  benchmark_dict['ts:run_start']  #计算总时间
    return copy_time, proc_time, totaltime

#图像预处理并推理
def infer_image(sess, image_files, config):
    input_details = sess.get_inputs()
    input_name = input_details[0].name
    floating_model = (input_details[0].type == 'tensor(float)')   # 判断是否为浮点模型
    height = input_details[0].shape[2]  #384
    width  = input_details[0].shape[3]  #128
    print(image_files)
    imgs=image_files
    img_bgr = cv2.imread(image_files)
    print("image size:", img_bgr.shape)
    img_bgr2 = cv2.resize(img_bgr, ( width,height))
    print("image resize:", img_bgr2.shape)
    img_rgb = img_bgr2[:,:,::-1]    #(384, 128, 3)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 预处理-归一化
    input_tensor = img_rgb / 255    # 预处理-构造输入 Tensor
    input_tensor = np.expand_dims(input_tensor, axis=0) # 加 batch 维度 (1, 384, 128, 3)
    input_tensor = input_tensor.transpose((0, 3, 1, 2)) # N, C, H, W
    input_tensor = np.ascontiguousarray(input_tensor)   # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
    input_tensor = torch.from_numpy(input_tensor).to(device).float() # 转 Pytorch Tensor
    input_data = input_tensor[:, :1, :, :]    #转单通道
    print(input_data.shape)

    #推理图片，计时
    start_time = time.time()  # 记录开始时间
    output = list(sess.run(None, {input_name: input_data.numpy()}))  # 进行推理并获取输出结果
    print("output.shape:", output[0].shape)
    stop_time = time.time()
    infer_time = stop_time - start_time  # 计算推理时间
    # 获取拷贝时间、子图处理时间和总时间
    copy_time, sub_graphs_proc_time, totaltime = get_benchmark_output(sess)
    proc_time = totaltime - copy_time

    return imgs, output, proc_time, sub_graphs_proc_time, height, width

#main 主程序####################################################################
def run_model(model, mIdx):
    print("\nRunning_Model : ", model, " \n")
    config = models_configs[model]
    # 将编译配置更新到 delegate_options 中
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)   
    #   拼接 "artifacts_folder" 的路径，将 model 名称添加到文件夹路径中
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/' 
    
    # delete the contents of this folder
    if args.compile or args.disable_offload:    # 如果命令行参数中有 --compile 或 --disable_offload
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    #编译和测试选不同的数据集
    if(args.compile == True):   # 如果参数中存在 --compile
        input_image = calib_images
        import onnx
        log = f'\nRunning shape inference on model {config["model_path"]} \n'
        print(log)
        onnx.shape_inference.infer_shapes_path(config['model_path'], config['model_path'])  # 根据校准图像执行形状推断
    else:
        input_image = test_images
    numFrames = config['num_images']
    if(args.compile):   # 如果 numFrames 大于校准帧数，则将其设置为校准帧数
        if numFrames > delegate_options['advanced_options:calibration_frames']:
            numFrames = delegate_options['advanced_options:calibration_frames']
    
    ############   set interpreter  ################################
    #根据不同的命令行参数选择不同的解释器
    if args.disable_offload : 
        EP_list = ['CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] , providers=EP_list,sess_options=so)
    elif args.compile:
        EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    else:
        EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    
    ############  run  session  ############################
    for i in range(len(input_image)):
        print("-----------image:", i, "-----------")
        input_images=input_image[i]
        # 运行推断函数，获取输出结果，处理时间和子图时间，以及高度和宽度
        imgs, output, proc_time, sub_graph_time, height, width  = infer_image(sess, input_images, config)
        # 计算总处理时间和子图时间
        total_proc_time = total_proc_time + proc_time if ('total_proc_time' in locals()) else proc_time
        sub_graphs_time = sub_graphs_time + sub_graph_time if ('sub_graphs_time' in locals()) else sub_graph_time
        total_proc_time = total_proc_time /1000000
        sub_graphs_time = sub_graphs_time/1000000

        # output post processing
        if(args.compile == False):  # post processing enabled only for inference, 如果不是编译模式，则执行后处理
            output = deploy_preprocess(output[0])   #获取推理结果并进行处理
            #print(output)
            pred_points = get_predicted_points(output[0])   #得到预测点位
            print(pred_points)
            eval_results = {}
            eval_results['pred_points'] = pred_points
            img = cv2.imread(input_images)  #导入图片用来画线
            img_plot = plot_slots(img, eval_results)    #画线
            cv2.imshow('XXX', img_plot)    #显示结果
            key = cv2.waitKey(1000) & 0xFF
            cv2.destroyAllWindows()
            save_path = os.path.join('../../../output_images','test_image'+str(i+1)+'.jpg') #保存路径
            cv2.imencode('.jpg', img_plot)[1].tofile(save_path)
        
        if args.compile or args.disable_offload :   # 如果是编译模式或者禁用了offload，则生成参数YAML文件
            gen_param_yaml(delegate_options['artifacts_folder'], config, int(height), int(width))
        log = f'\n \nCompleted_Model : {mIdx+1:5d}, Name : {model:50s}, Total time : {total_proc_time/(i+1):10.2f}, Offload Time : {sub_graphs_time/(i+1):10.2f} , DDR RW MBs : 0\n \n ' #{classes} \n \n'
        print(log)  # 打印日志信息
        if ncpus > 1:   # 如果使用了多个CPU，则释放信号量
            sem.release()

models = ['XXX_yolox']
log = f'\nRunning {len(models)} Models - {models}\n'
print(log)

#以下为线程控制，由此处进入运行程序>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def join_one(nthreads): # 定义一个函数来加入一个线程
    global run_count
    sem.acquire()     # 获取一个信号量，控制线程同步
    run_count = run_count + 1   # 增加运行计数
    return nthreads - 1 # 返回线程数减1

def spawn_one(models, idx, nthreads):   # 定义一个函数来创建并启动一个线程
    # 创建一个新的进程，目标函数是 run_model，参数是 models 和 idx
    p = multiprocessing.Process(target=run_model, args=(models,idx,))
    p.start()   # 启动进程
    return idx + 1, nthreads + 1    # 返回新的 idx 和 nthreads

if ncpus > 1:   # 如果有多个CPU，则创建并启动多个线程
    for t in range(min(len(models), ncpus)):
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    while idx < len(models):     # 当还有未处理的 model 时, 等待一个线程完成，并减少线程数
        nthreads = join_one(nthreads)
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    for n in range(nthreads):
        nthreads = join_one(nthreads)
else : #如果只有一个CPU：使用一个循环顺序地处理每个模型。每个模型会直接调用run_model函数进行处理。
    for mIdx, model in enumerate(models):
        run_model(model, mIdx)
```
</details>


### model-artifacts
分析编译深度学习模型后生成的文件：
```py
└── model-artifacts         #文件都是以最后的输出层命名，分为四块网络结构
    ├── 1102_tidl_io_1.bin  #io配置文件
    ├── 1102_tidl_net.bin   #网络模型的二进制文件
    ├── allowedNode.txt     #允许的节点列表文件
    ├── onnxrtMetaData.txt  #ONNX运行时的元数据文件
    ├── param.yaml          #参数配置文件
    ├── XXX_yolox.onnx      #深度学习模型的原始ONNX文件
    └── tempDir             #模型编译过程的临时文件和输出文件
        ├── 1102_calib_raw_data.bin #用于校准的原始数据文件
        ├── 1102_tidl_io_1.bin      #输入数据的二进制文件
        ├── 1102_tidl_io__LayerPerChannelMean.bin   #存储每个通道的平均值的二进制文件。对于量化和归一化操作，需要存储每个通道的平均值。
        ├── 1102_tidl_io_.perf_sim_config.txt   #性能模拟的配置文件
        ├── 1102_tidl_io_.qunat_stats_config.txt    #量化统计的配置文件
        ├── 1102_tidl_io__stats_tool_out.bin    #输出二进制文件。用于存储进行量化统计时的一些中间结果。
        ├── 1102_tidl_net       #编译后的深度学习模型相关文件
        │   ├── bufinfolog.csv  #缓冲区信息的CSV文件，可能包含模型各个层的输入和输出缓冲区的大小和信息。
        │   ├── bufinfolog.txt  #缓冲区信息的文本文件
        │   └── perfSimInfo.bin #性能模拟信息的二进制文件。可能包含模型在性能模拟时的一些统计数据。
        ├── 1102_tidl_net.bin
        ├── 1102_tidl_net.bin_netLog.txt        #模型编译日志的文本文件
        │   ├── #TIDL Layer Name, Out Data Name, Group, #Ins, #Outs
        │   ├── #Inbuf Ids, Outbuf Id: 输入输出缓冲区的标识符， In NCHW, Out NCHW: 输入输出数据的格式和维度信息
        │   └── #MACS: 模型在推理过程中进行的乘加运算，用于衡量模型的计算量和复杂度。
        ├── 1102_tidl_net.bin_paramDebug.csv    #包含模型参数的调试信息的CSV文件。记录了每个层量化前后的参数差异，模型通常以浮点数形式进行训练，量化通常将浮点参数转换为固定位数的整数参数。
        │   ├── #meanDifference: 参数的平均差异，maxDifference: 参数的最大差异，
        │   ├── #meanOrigFloat: 原始浮点参数的平均值，meanRelDifference: 参数的相对平均差异，
        │   ├── #orgmax: 原始浮点参数的最大值，quantizedMax: 量化后参数的最大值
        │   ├── #orgAtmaxDiff: 原始浮点参数在最大值处的差异，quantizedAtMaxDiff: 量化后参数在最大值处的差异，maxRelDifference: 参数的最大相对差异
        │   └── #Scale: 参数的缩放比例，在量化中，使用缩放因子将浮点参数映射到整数参数；
        ├── 1102_tidl_net.bin.layer_info.txt    #包含模型各个层信息的文本文件
        ├── 1102_tidl_net.bin.svg   #该部分模型结构的可视化图像文件
        ├── graphvizInfo.txt    #模型结构的图形化文本信息
        └── runtimes_visualization.svg  #整个网络结构可视化文件
```

**为什么有些网络结构编译后被拆分成了多组不同的二进制文件？**（*4 subgraph output nodes*）: 多网络结构文件拼接成一个完整的网络，但由于不支持的层被offload到arm端运行，因此在相应的位置被拆分，前期网络结构设计时需要尽量避免出现该情况。



# TIDL tools c++推理(ongoing)
TIDL runtime 提供的CPP api解决方案仅支持模型推理，因此仍需在PC上运行Python示例以生成模型工件。
[edgeai-tidl-tools/examples/osrt_cpp](https://github.com/TexasInstruments/edgeai-tidl-tools/tree/master/examples/osrt_cpp)
```sh
export SOC=am68pa
mkdir build2 && cd build2
cmake -DFLAG1=val -DFLAG2=val ../../../examples


ongoing....
```

# SK板运行自定义深度学习模型(ongoing)
通过SD卡配置编译生成的模型：
> 配置模型文件夹 custom_model 放入/opt/modelzoo文件夹
>> artifacts：存放编译生成的工件，model-artifacts
model：原onnx模型，.onnx (.prototxt)
param.yaml：配置文件, 其中需要修改model_path等参数 (以modelzoo中例程的param为基准，参照model-artifacts中生成的param修改参数)
(dataset.yaml：数据集类别对应文件)

通过SD卡配置`/opt/edgeai-gst-apps/configs/XXX.yaml`，在model参数中索引上面建立的模型文件夹 custom_model, 并根据size修改输入输出，分辨率size一定要改好，否则很容易报错
```sh
#通过minicom连接串口
sudo minicom -D /dev/ttyUSB2 -c on
root #登录
#运行自定义实例
cd /opt/edgeai-gst-apps/apps_cpp
./bin/Release/app_edgeai ../configs/XXX.yaml
#Ctrl+C 安全退出
```

如果不是常规的OD、SEG等任务，需要高度自定义的话，需要修改SK板 `/opt/edgeai-gst-apps` DEMO相关的源码，主要阅读源码并参考两大文档：
[Edge AI sample apps &mdash; Processor SDK Linux for SK-TDA4VM Documentation](https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-edgeai/TDA4VM/08_06_01/exports/docs/common/sample_apps.html)
[Running Simple demos &mdash; Processor SDK Linux for Edge AI Documentation](https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-sk-tda4vm/latest/exports/docs/running_simple_demos.html)
下面对demo源码进行研读：

---
也许可以将tidl-tools放到板子里然后运行? 然后选择正确的平台
/opt/vision_apps/vx_app_arm_remote_log.out 查arm log的脚本






























---
YOLO-pose实例：[Practicing Yoga with AI: Human Pose Estimation on the TDA4VM](https://www.hackster.io/whitney-knitter/practicing-yoga-with-ai-human-pose-estimation-on-the-tda4vm-fe2549?auth_token=68e0af8f809985238fdb2b7554c48a46)
官方视频：[Efficient object detection using Yolov5 and TDA4x processors | Video | TI.com](https://www.ti.com/video/6286792047001)
官方文档：[4. Deep learning models &mdash; Processor SDK Linux for SK-TDA4VM Documentation](https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-edgeai/TDA4VM/08_06_01/exports/docs/common/inference_models.html)
> TDA4系列文章：
[TDA4①：SDK, TIDL, OpenVX](https://wangyujie.space/TDA4VM/)
[TDA4②：环境搭建、模型转换、Demo及Tools](https://wangyujie.space/TDA4VM2/)
[TDA4③：YOLOX的模型转换与SK板端运行](https://wangyujie.space/TDA4VM3/)
[TDA4④：部署自定义深度学习模型](https://wangyujie.space/TDA4VM4/)