
# Pytorch Deep learning Notes
# ----PyTorch 深度学习学习笔记（新手可轻松复现版）

[![GitHub stars](https://img.shields.io/github/stars/Jackksonns/pytorch-deep-learning-notes)](https://github.com/Jackksonns/pytorch-deep-learning-notes/stargazers)
 [![GitHub forks](https://img.shields.io/github/forks/Jackksonns/pytorch-deep-learning-notes)](https://github.com/Jackksonns/pytorch-deep-learning-notes/network/members)
 [![GitHub issues](https://img.shields.io/github/issues/Jackksonns/pytorch-deep-learning-notes)](https://github.com/Jackksonns/pytorch-deep-learning-notes/issues)
 [![GitHub license](https://img.shields.io/github/license/Jackksonns/pytorch-deep-learning-notes)](https://github.com/Jackksonns/pytorch-deep-learning-notes/blob/main/LICENSE)

------

## 项目概述

本仓库汇集了较为系统的 PyTorch 深度学习学习笔记与实现，包含较为完整的从基础神经网络到进阶架构与实战任务。代码参考并改进自李沐（Li Mu）的课程与小土堆的 PyTorch 教程，力求实现既简洁清晰又能直接运行的示例，便于学习与复现。

## 目录

```bash
总深度学习学习笔记/
├── deep_learning_basic/           # PyTorch 基础概念与操作
├── pytorch完整流程手搓basis/       # 完整的 PyTorch 训练流程示例（引自小土堆老师）
├── 经典网络架构/                  # 常见网络实现（LeNet/ResNet 等）
│   ├── LeNet/
│   ├── AlexNet/
│   ├── VGG/
│   ├── ResNet/
│   ├── GoogLeNet/
│   └── NiN/
├── 图像分类CIFAR-10/               # CIFAR-10 相关实现
├── 树叶分类Classify_leaves/        # 比赛与项目实现
├── 狗品种识别ImageNet_Dogs/        # 狗品种识别任务
├── 目标检测数据集及代码/            # 目标检测相关代码与数据
├── Bounding_Box/                   # 边界框检测实现
├── 风格迁移/                       # 风格迁移示例
├── fine_tuning/                    # 迁移学习与微调示例
├── bert/                           # BERT 预训练与微调
├── transformer/                    # Transformer 实现
├── SSD/                            # Single Shot MultiBox Detector
├── Semantic_Segmentation_and_Dataset/  # 语义分割与数据集
├── FCN/                            # 全卷积网络实现
└── 房价预测House_price_predict/     # 房价回归示例
```

## 🚀 快速上手

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择合适的安装命令）
pip install torch torchvision torchaudio

# 额外依赖
pip install matplotlib numpy pandas scikit-learn tensorboard
```

### 使用示例

1. **从基础开始**：

   ```bash
   cd deep_learning_basic/
   python custom_layer.py
   ```

2. **完整训练流程演示**：

   ```bash
   cd pytorch完整流程手搓basis/
   python train.py
   ```

3. **经典网络运行示例**：

   ```bash
   cd 经典网络架构/LeNet/
   python LeNet.py
   ```

------

## 主要学习内容

### 1. 基础概念

- 自定义层与模块（Custom Layers）
- 图像增强与数据扩充（Image Augmentation）——数据处理
- 从零构建模型（Model Construction）——pytorch代码基础
- 参数管理（权重初始化、优化器Adam/AdamW选择等）

### 2. 经典网络架构

- LeNet-5：早期卷积神经网用于手写数字识别（经典中的经典）
- AlexNet：在 ImageNet 上的突破性成果（性能突破的里程碑）
- VGG：以堆叠小卷积核著称的深度网络
- ResNet：残差学习，解决深层网络退化问题（可训练出千层模型）
- GoogLeNet：Inception 结构堆叠
- NiN（Network in Network）：局部非线性映射思想

### 3. 计算机视觉应用

- 图像分类（CIFAR-10、树叶分类、狗品种识别）
- 目标检测（SSD、边界框检测）
- 语义分割（FCN、VOC 数据集）
- 风格迁移（Neural Style Transfer）

### 4. 自然语言处理

- BERT：预训练与下游微调
- Transformer：注意力机制的实现与应用
- 编码器-解码器（Encoder-Decoder）序列建模

### 5. 实战应用

- 房价预测：使用 MLP 做回归任务
- 迁移学习：利用预训练模型做微调提升效果
- 模型优化：训练流程优化（学习率调度、正则化等）

------

## Star & Fork this repo and Learn, 你将会收获：

### 技能层面

- PyTorch 框架使用
- 设计与实现神经网络架构
- 搭建并运行训练流水线
- 数据预处理与增强技巧
- 模型评估与性能优化

### 实战经验

- 真实数据集的下载、预处理与训练
- 竞赛级别的实现思路与工程化细节
- 迁移学习在少样本任务中的应用
- 模型部署与推理优化的基础思考

------

## 代码质量特点

- **注释详尽**：代码包含中英文注释，便于快速阅读与理解
- **模块化设计**：清晰、可复用的代码结构
- **鲁棒性考虑**：包含基础的错误检测与处理逻辑
- **训练监控**：支持训练过程可视化（如 TensorBoard）
- **遵循最佳实践**：对齐 PyTorch 与深度学习社区常见规范

------

## 具体实验

### 分类任务

- **CIFAR-10**：使用自定义架构达到有竞争力的准确率
- **树叶分类**：实现并比较 ResNet18/ResNet50 与自定义网络
- **狗品种识别**：通过迁移学习显著提升效果

### 检测任务

- **目标检测**：实现 SSD，支持自定义数据集训练
- **边界框检测**：包括基于锚框的检测实现与示例

### NLP 任务

- **BERT 微调**：在自定义数据集上成功微调并获得良好表现
- **Transformer 实现**：包含注意力机制与序列建模示例

# 学习总结

经过系统学习与大量练习，我把从基础概念、经典网络到实战任务的理解和实现整理成了这个仓库。写这些笔记的初衷是：让自己把知识吃透，并把能直接运行、容易复现的代码分享给更多像我一样的学习者。

如果你正在学习深度学习或准备做项目／竞赛，这个仓库可以作为一个动手实践的起点：

- 每个模块尽量提供可运行的示例和详尽注释，便于快速上手；
- 代码风格保持模块化，方便拿来微调或二次开发；
- 我也欢迎大家提出 issues、PR 或直接在 Discussions 里交流问题与改进建议。

希望你能拿走你需要的代码、跑通示例、并把你的经验回馈到仓库中——让它变得更好，也让更多人能更容易地上手深度学习。欢迎大家一起用、一起改、一起交流一起进步呀！

## To-do List

-  增加更多前沿架构（如 Swin Transformer）
-  实现更多 NLP 模型（GPT、BART、T5）
-  补充强化学习示例

## 🙏 致谢

感谢以下开源资源与社区对本项目的启发与支持：

- **李沐（Li Mu）老师**：优秀的《动手学深度学习》（Dive into Deep Learning）课程与 d2l 包
- **小土堆（Xiao Tu Dui）老师**：详尽的 PyTorch 教程资源和易上手的代码资源
- **PyTorch 社区**：提供强大的深度学习框架与生态
- **广泛的开源社区**：感谢大家共享的资料与工具（本项目也包含一些在kaggle比赛中表现优异的代码实现，具体引用详见对应文件，感谢大家的开源支持！）

## ⭐ 如果你觉得这个仓库对你有帮助，请点个 Star！你的支持是我持续更新与分享的动力。

**Contact me**: [Jackksonns (Jackson KK)](https://github.com/Jackksonns).

