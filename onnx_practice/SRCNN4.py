import torch
import torch.onnx
# 定义一个继承自torch.nn.Module的模型类
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

# 导入torch.onnx.symbolic_registry中的register_op函数
from torch.onnx.symbolic_registry import register_op

# 定义asinh的符号化函数
def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

# 注册asinh操作
register_op('asinh', asinh_symbolic, '', 9)

# 创建模型实例
model = Model()

# 创建输入数据
input = torch.rand(1, 3, 10, 10)

# 导出模型为ONNX格式
torch.onnx.export(model, input, 'asinh.onnx')