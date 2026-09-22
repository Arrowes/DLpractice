import onnx
import onnxruntime
import torch
from torch import nn
from torch.nn.functional import interpolate
import torch.onnx
import cv2
import numpy as np


class NewInterpolate(torch.autograd.Function):
    # 自定义的插值算子，继承自torch.autograd.Function
    @staticmethod
    def symbolic(g, input, scales):
        # 静态方法，用于定义符号图的构建过程, g: 符号图构建器, input: 输入张量, scales: 缩放因子

        return g.op("Resize",  # 使用Resize操作
                    input,  # 输入张量
                    g.op("Constant", value_t=torch.tensor([], dtype=torch.float32)),  # 空的常量张量
                    scales,  # 缩放因子
                    coordinate_transformation_mode_s="pytorch_half_pixel",  # 坐标转换模式为pytorch_half_pixel
                    cubic_coeff_a_f=-0.75,  # cubic插值的系数a为-0.75
                    mode_s='cubic',  # 插值模式为cubic
                    nearest_mode_s="floor")  # 最近邻插值模式为floor

    @staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(input,
                           scale_factor=scales,
                           mode='bicubic',
                           align_corners=False)


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
