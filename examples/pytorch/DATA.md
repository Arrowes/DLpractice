# 数据与训练产物

本目录中的示例来自不同学习阶段，所需数据和路径并不完全统一。

## 数据集

- CIFAR-10 示例可使用 `torchvision.datasets.CIFAR10(..., download=True)` 自动下载。
- 其他示例若引用 `data/` 或 `dataset/`，请将相应数据放在脚本预期的相对目录中。
- 数据集目录已加入 `.gitignore`，不会随普通提交上传。

## 模型与日志

以下内容属于可重新生成的实验产物，默认只保存在本机：

- PyTorch 权重：`*.pth`、`*.pt`
- ONNX 模型：`*.onnx`
- TensorBoard 日志：`logs/`、`logs_*`、`runs/`
- 检查点与权重目录：`checkpoints/`、`weights/`

旧脚本可能使用固定相对路径。运行前先从脚本所在目录确认路径，并避免一次性执行整个目录。
