# 深度学习学习与实践

本仓库整理了深度学习基础笔记、PyTorch 示例、部署练习和目标检测参考资料。根目录只保留导航与环境说明，详细内容按用途放在独立目录中。

## 内容导航

- [深度学习概念笔记](docs/deep-learning-concepts.md)：从神经网络基础到 CNN、目标检测和序列模型的系统笔记。
- [PDF 转写原稿](docs/source-notes/original-deep-learning-notes.md)：从个人 PDF 笔记整理出的 Markdown 版本。
- [PyTorch 示例](examples/pytorch/)：分类、卷积网络、循环网络、生成模型等练习代码。
- [SRCNN 部署练习](projects/srcnn-deployment/)：模型导出和 ONNX Runtime 推理示例。
- [YOLO 参考资料](references/yolo/)：目标检测相关笔记与资料。
- [延伸阅读](references/reading/)：课程 PDF、面试资料和演示文稿。

对应博客：

- [深度学习基础](https://wangyujie.space/DL/)
- [模型训练实践](https://wangyujie.space/DLtrain/)
- [模型部署实践](https://wangyujie.space/DLdeploy/)

## 目录结构

```text
.
├── docs/                  # 概念笔记与原始转写
├── examples/pytorch/      # PyTorch 学习代码
├── projects/              # 可独立运行的实践项目
└── references/            # YOLO 与延伸阅读资料
```

## 环境准备

建议为本仓库创建独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch 是否使用 CUDA 取决于显卡、驱动和 CUDA 版本。需要 GPU 时，请按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择匹配的安装命令，再安装其余依赖。

## 数据与模型文件

数据集、训练日志、模型权重和导出的 ONNX 文件不再纳入 Git。现有文件仍保留在本地原位置；新环境可根据具体示例重新下载数据或运行训练、导出流程生成产物。详见 [数据说明](examples/pytorch/DATA.md)。

## 使用建议

1. 先阅读概念笔记建立知识框架。
2. 在 `examples/pytorch/` 中按主题运行小型示例。
3. 使用 `projects/srcnn-deployment/` 学习模型导出和推理。
4. 将实验数据、权重和日志保留在本地，避免仓库体积继续增长。

> 本仓库包含课程练习和第三方参考材料。版权归原作者所有；公开复用前请核对各资料的许可范围。
