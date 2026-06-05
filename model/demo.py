"""MiniGPT 快速训练 Demo - 验证模型能否正常工作"""
import torch
import torch.optim as optim
from model import mini_gpt

# ============================================================
# 1. 准备数据：一段中文文本，字符级分词
# ============================================================
text = """
人工智能太好玩了！我想用PyTorch从零手搓一个大模型。
大模型其实就是一个文字接龙的游戏。给它前面的词，它就能猜出后面的词。
只要算力足够，简单的框架也能涌现出惊人的智慧。
冲鸭，打败ChatGPT就靠这十天了！
深度学习让机器能够理解语言，这真是太神奇了。
神经网络模仿人脑的工作方式，一层一层地提取特征。
注意力机制让模型知道哪些信息是重要的。
Transformer架构改变了整个自然语言处理领域。
预训练加微调的范式让AI变得更加强大。
从GPT到Claude，大语言模型的发展日新月异。
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# ============================================================
# 2. 超参数
# ============================================================
batch_size = 8
ctx_len = 64
d_model = 64
num_heads = 2
d_ff = 256
num_layers = 2
epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"设备: {device} | 词表大小: {vocab_size} | 数据长度: {len(data)}")

# ============================================================
# 3. 数据加载
# ============================================================
def get_batch():
    ix = torch.randint(len(data) - ctx_len, (batch_size,))
    x = torch.stack([data[i : i + ctx_len] for i in ix])
    y = torch.stack([data[i + 1 : i + ctx_len + 1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================
# 4. 建模型
# ============================================================
model = mini_gpt(
    vocab_size=vocab_size,
    d_model=d_model,
    max_seq_len=ctx_len,
    n_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
).to(device)

opt = optim.AdamW(model.parameters(), lr=1e-3)

param_count = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {param_count:,}")

# ============================================================
# 5. 训练
# ============================================================
model.train()
print("\n开始训练...")

for epoch in range(1, epochs + 1):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# ============================================================
# 6. 生成测试
# ============================================================
model.eval()
prompt = "人工智能"
input_ids = torch.tensor([[char_to_idx[ch] for ch in prompt]], dtype=torch.long, device=device)
generated_text = prompt

with torch.no_grad():
    for _ in range(60):
        input_cond = input_ids[:, -ctx_len:]
        logits, _ = model(input_cond)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        generated_text += idx_to_char[next_token.item()]

print(f"\n生成结果:\n{generated_text}")
