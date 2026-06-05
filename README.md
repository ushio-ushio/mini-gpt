# MiniGPT

MiniGPT 是一个基于 PyTorch 从零实现的精简版 GPT（Generative Pre-trained Transformer）架构大语言模型。本项目不仅包含基础模型的构建与训练，还涵盖了从极简测试、数据集处理到 FastAPI 模型部署，以及结合 LangGraph 构建 Agent（智能体）的完整生态。非常适合用于学习大模型底层原理与工程落地。

## 模型框架

MiniGPT 采用了标准的 **Decoder-Only Transformer** 架构，即自回归语言模型结构，主要包含以下核心组件：

- **自注意力机制 (Self-Attention)**：实现了带掩码（Causal Mask）的多头自注意力机制，保证模型在预测下一个 Token 时只能看到当前和之前的上下文。
- **前馈神经网络 (FeedForward)**：两层全连接网络加上 非线性激活函数 和 Dropout，增加模型的非线性拟合能力。
- **解码器层 (DecoderBlock)**：堆叠了多个标准的 Transformer Decoder Block，内部采用 `Pre-LayerNorm` 结构（即先做 LayerNorm 再进入 Attention 和 FeedForward），并借助残差连接（Residual Connections）提升深层网络训练的稳定性。
- **混合嵌入 (Embeddings)**：包含词嵌入（Token Embedding）以及绝对位置嵌入（Position Embedding），用于将输入序列转化为稠密向量并赋予其序列位置信息。

## 项目结构

### 核心模型与训练

- **`model.py`**: MiniGPT 模型核心实现，包括多头自注意力机制（Self-Attention）、前馈神经网络（FeedForward）、解码器块（DecoderBlock）等。
- **`tokenizer.py`**: 字符级/字节级分词器（ByteTokenizer），负责文本的编码与解码。
- **`dataset.py`**: 语言模型数据加载器（LMDataset），将文本数据预处理为模型所需的上下文张量。
- **`train.py`**: 完整的训练脚本，支持混合精度训练（AMP）、梯度累加、梯度裁剪等进阶特性。训练权重保存在 `checkpoints/` 目录下。
- **`demo.py`**: 快速验证脚本。在一小段中文文本上进行快速训练（过拟合）并生成文本，用于验证模型结构是否正常工作。
- **`download_data.py`**: 训练数据准备脚本，用于解压 WikiText-103 数据集并生成 `train.txt` 和 `val.txt`。

### Agent 生态 (`agent/` 目录)

- **`api.py`**: FastAPI 后端服务。自动加载 `checkpoints/pretrain.pt` 权重，并通过端点对外提供文本生成服务（`/generate`）。
- **`llm.py`**: 基于 LangChain `BaseLLM` 封装的本地模型桥梁（`LocalDecoderLLM`），能让 LangChain 节点透明地调用本地部署的 API。
- **`agent.py`**: 基于 LangGraph 的智能体工作流。实现了一个具有主动辅导与工具调用能力的 Agent（不仅能读取本地文档，还能根据材料自动向用户反向提出启发式学习考核问题）。

## 快速开始

### 1. 验证模型 (Sanity Check)

在训练完整模型前，建议先运行 `demo.py` 查看模型推理是否正常（需要约 1-2 分钟）：

```bash
python demo.py
```

该文件会训练一个小型网络并试图输出一串中文文本。

### 2. 数据处理与准备大模型训练

准备 WikiText-103 语料，将其 zip 压缩包放入 `data/` 目录后运行：

```bash
python download_data.py
```

运行训练流程（支持 CUDA 自动加速）：

```bash
python train.py
```

训练结束后，模型权重将被保存为 `checkpoints/pretrain.pt`。

### 3. 本地模型 API 部署

当模型训练完毕后，可启动推理 API 以提供模型服务：

```bash
cd agent
python api.py
```

API 启动后，将监听在 `http://127.0.0.1:8000`。

### 4. 运行 LangGraph Agent

保持 `api.py` 的服务在后台运行，然后打开一个新的终端执行：

```bash
cd agent
python agent.py
```

你将看到控制台打印智能体的思考过程、工具调用结果（如调用 `read_local_document` 读取学习资料），以及 Agent 根据文档内容向用户主动提出的考核问题。

## 🛠 技术栈

- **核心框架**: PyTorch
- **API 服务**: FastAPI, Uvicorn
- **Agent/LLM Ops**: LangChain, LangGraph
