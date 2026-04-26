
好，我们开始看 `model.py`。这个文件你要把它理解成：

```text
nanoGPT 的“模型本体文件”
```

它不负责训练循环，不负责数据读取，不负责命令行参数，主要负责：

```text
1. 定义 GPT 模型结构
2. 定义 forward
3. 定义 generate
4. 定义如何加载 GPT-2 预训练权重
5. 定义 optimizer 参数分组
6. 定义一些工程辅助函数
```

文件开头也写了：这是一个完整 GPT Language Model 的定义，全部放在这个单文件里，并参考了 OpenAI GPT-2 TensorFlow 实现和 Hugging Face GPT-2 PyTorch 实现。

---

# 0. 先给你一张总地图

`model.py` 里面主要有这些类和函数：

```text
LayerNorm
↓
CausalSelfAttention
↓
MLP
↓
Block
↓
GPTConfig
↓
GPT
    ├── get_num_params()
    ├── _init_weights()
    ├── forward()
    ├── crop_block_size()
    ├── from_pretrained()
    ├── configure_optimizers()
    ├── estimate_mfu()
    └── generate()
```

你可以这样理解它们之间的关系：

```text
GPT
├── token embedding
├── position embedding
├── 多个 Block
│   ├── LayerNorm
│   ├── CausalSelfAttention
│   ├── LayerNorm
│   └── MLP
├── final LayerNorm
└── lm_head
```

也就是你自己写过的 MiniGPT 的工程版。

---

# 1. 文件开头的 import

```python
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
```

这里你要知道几个点。

## `math`

主要用于 attention 里面的缩放：

```python
1.0 / math.sqrt(k.size(-1))
```

这对应你熟悉的：

```text
QK^T / sqrt(d_k)
```

---

## `inspect`

这个你之前 toy GPT 里可能没用过。

它后面用于判断当前 PyTorch 的 `AdamW` 是否支持 `fused=True`：

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
```

这属于工程优化。

你可以暂时理解成：

```text
如果当前环境支持更快的 fused AdamW，就用；
不支持就普通 AdamW。
```

---

## `dataclass`

用于定义 `GPTConfig`。

这个比普通 class 简洁，适合保存配置。

你自己的 miniGPT 里可能是很多变量：

```python
block_size = 128
vocab_size = ...
n_embd = 256
n_head = 4
n_layer = 4
```

nanoGPT 会集中成：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

这个后面我们再细看。

---

# 2. `LayerNorm`

源码核心是：

```python
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

这段看起来像是在重复 PyTorch 的 `nn.LayerNorm`，但它有一个目的：

```text
允许 LayerNorm 不使用 bias。
```

PyTorch 的标准 `nn.LayerNorm` 以前不太方便直接设置 `bias=False`。所以 nanoGPT 自己写了一个。

---

## 你要对照自己的 miniGPT

你可能写过：

```python
self.ln1 = nn.LayerNorm(n_embd)
self.ln2 = nn.LayerNorm(n_embd)
```

nanoGPT 写成：

```python
self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
```

本质一样：

```text
都是对最后一个 embedding 维度做归一化。
```

如果输入是：

```text
x.shape = (B, T, C)
```

其中：

```text
B = batch size
T = sequence length
C = n_embd
```

LayerNorm 是对每个 token 的 `C` 维向量做归一化。

---

## 为什么需要 LayerNorm？

Transformer 训练时，每层经过 Attention、MLP、Residual 之后，数值分布会变化。LayerNorm 让每个 token 的 hidden state 更稳定。

你先记：

```text
LayerNorm 是训练稳定器。
```

---

# 3. `CausalSelfAttention`

这是整个文件里最核心的部分之一。

源码结构是：

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        ...

    def forward(self, x):
        ...
```

它对应你自己 miniGPT 里的：

```text
masked multi-head self-attention
```

---

# 3.1 初始化部分

核心代码：

```python
assert config.n_embd % config.n_head == 0
```

这句话非常重要。

意思是：

```text
embedding 维度必须能被 head 数整除。
```

比如 GPT-2 small：

```text
n_embd = 768
n_head = 12
head_size = 768 / 12 = 64
```

每个 head 负责 64 维。

---

## QKV 合并

nanoGPT 写的是：

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```

你自己的 toy 版本可能是：

```python
self.query = nn.Linear(n_embd, n_embd)
self.key = nn.Linear(n_embd, n_embd)
self.value = nn.Linear(n_embd, n_embd)
```

nanoGPT 把这三个合成一个：

```text
一次 Linear 输出 Q、K、V 三份结果。
```

也就是：

```text
输入维度：n_embd
输出维度：3 * n_embd
```

然后在 forward 里再切开：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

这就是工程化处理。

你的版本：

```text
Linear Q
Linear K
Linear V
```

nanoGPT：

```text
一个大 Linear，同时算 QKV
```

数学上没变，但效率更好，代码也更紧凑。

---

## 输出投影

```python
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
```

多头 attention 拼接回来之后，还要经过一个线性层。

也就是：

```text
multi-head 输出
↓
output projection
↓
回到 n_embd 维
```

这和标准 Transformer 一样。

---

## 两个 dropout

```python
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
```

这里有两个 dropout：

```text
attn_dropout：
作用在 attention 权重上。

resid_dropout：
作用在 attention 输出投影之后。
```

也就是：

```text
softmax 后的注意力矩阵可以 dropout
最终输出也可以 dropout
```

不过如果使用 PyTorch 的 Flash Attention 分支，attention dropout 会通过 `scaled_dot_product_attention` 的参数传进去。

---

## Flash Attention 判断

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
```

如果 PyTorch 里有 `scaled_dot_product_attention`，就用更快的实现。

如果没有，就走手写 attention。

这说明 nanoGPT 同时保留了两种路径：

```text
高效路径：PyTorch 内置 scaled_dot_product_attention
普通路径：手写 QK^T + mask + softmax + V
```

对你学习来说，重点看普通路径，因为它和你手搓版本最接近。

---

## causal mask

如果没有 Flash Attention，代码会注册一个 mask：

```python
self.register_buffer(
    "bias",
    torch.tril(torch.ones(config.block_size, config.block_size))
        .view(1, 1, config.block_size, config.block_size)
)
```

这个 mask 是下三角矩阵。

例如 `T=5`：

```text
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

意思是：

```text
第 0 个 token 只能看自己
第 1 个 token 可以看 0、1
第 2 个 token 可以看 0、1、2
...
```

也就是 causal language model 的核心：

```text
当前位置不能偷看未来 token。
```

---

# 3.2 forward 部分

输入：

```python
B, T, C = x.size()
```

这里：

```text
B = batch size
T = 当前序列长度
C = embedding dimension，也就是 n_embd
```

例如：

```text
B = 32
T = 256
C = 384
```

---

## 先得到 Q、K、V

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

如果：

```text
x.shape = (B, T, C)
```

那么：

```text
self.c_attn(x).shape = (B, T, 3C)
```

split 之后：

```text
q.shape = (B, T, C)
k.shape = (B, T, C)
v.shape = (B, T, C)
```

---

## reshape 成多头格式

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

这一步非常重要。

假设：

```text
B = 32
T = 256
C = 384
n_head = 6
head_size = 64
```

原来：

```text
q.shape = (32, 256, 384)
```

先变成：

```text
(32, 256, 6, 64)
```

再 transpose：

```text
(32, 6, 256, 64)
```

也就是：

```text
(B, n_head, T, head_size)
```

为什么要这样？

因为每个 head 都要独立做 attention。

---

## 如果使用 Flash Attention

```python
y = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.dropout if self.training else 0,
    is_causal=True
)
```

这里 `is_causal=True` 等价于自动加 causal mask。

你可以把它理解成 PyTorch 替你做了：

```text
QK^T / sqrt(d_k)
+ causal mask
+ softmax
+ dropout
+ V
```

---

## 如果不用 Flash Attention：手写 attention

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

形状变化：

```text
q.shape = (B, nh, T, hs)
k.transpose(-2, -1).shape = (B, nh, hs, T)

att.shape = (B, nh, T, T)
```

`att[i, head, t1, t2]` 表示：

```text
第 i 个样本中，
某个 head 下，
位置 t1 对位置 t2 的注意力分数。
```

---

## 加 causal mask

```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

未来位置填成负无穷。

为什么是 `-inf`？

因为后面 softmax：

```python
att = F.softmax(att, dim=-1)
```

`softmax(-inf)` 约等于 0。

所以未来 token 的注意力概率变成 0。

---

## attention dropout

```python
att = self.attn_dropout(att)
```

这一步是对注意力概率做 dropout。

---

## 乘以 V

```python
y = att @ v
```

形状：

```text
att.shape = (B, nh, T, T)
v.shape   = (B, nh, T, hs)

y.shape   = (B, nh, T, hs)
```

这就是每个 head 的输出。

---

## 多头拼回去

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

从：

```text
(B, nh, T, hs)
```

变成：

```text
(B, T, nh, hs)
```

再拼成：

```text
(B, T, C)
```

---

## 输出 projection

```python
y = self.resid_dropout(self.c_proj(y))
```

也就是：

```text
多头拼接结果
↓
线性投影
↓
dropout
↓
返回
```

---

# 3.3 你应该如何对照自己的 Attention？

你可以这样对照：

|你的 miniGPT|nanoGPT|
|---|---|
|`query = Linear(...)`|`c_attn` 一次算 QKV|
|`key = Linear(...)`|`c_attn` 一次算 QKV|
|`value = Linear(...)`|`c_attn` 一次算 QKV|
|手写 mask|支持 Flash Attention，否则手写 mask|
|手动 tril mask|`register_buffer("bias", ...)`|
|拼接 head|`transpose + contiguous + view`|
|输出 Linear|`c_proj`|

这部分最重要的结论是：

```text
nanoGPT 的 attention 数学原理和你写的一样；
区别在于它把 QKV 合并、支持高效 attention、mask 做成 buffer。
```

---

# 4. `MLP`

源码核心：

```python
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

这就是 Transformer 里的前馈网络。

结构是：

```text
n_embd
↓
4 * n_embd
↓
GELU
↓
n_embd
↓
dropout
```

比如：

```text
n_embd = 768
```

那么 MLP 中间层是：

```text
3072
```

这就是 GPT-2 里的常见设置。

---

## 对照你的版本

你可能写过：

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.GELU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout),
)
```

nanoGPT 只是拆开写成：

```python
self.c_fc
self.gelu
self.c_proj
self.dropout
```

本质一样。

---

## 这里有什么工程点？

主要有两个：

```text
1. bias 是否使用由 config.bias 控制
2. dropout 由 config.dropout 控制
```

也就是说，模型结构不是写死的，而是配置控制。

---

# 5. `Block`

源码：

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

这是一个完整 Transformer block。

结构是：

```text
x
↓
LayerNorm
↓
CausalSelfAttention
↓
Residual Add

↓
LayerNorm
↓
MLP
↓
Residual Add
```

公式就是：

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

---

## 这是 Pre-LN Transformer

注意它是：

```python
x + self.attn(self.ln_1(x))
```

不是：

```python
self.ln_1(x + self.attn(x))
```

也就是说，它是：

```text
先 LayerNorm，再 Attention，最后残差相加。
```

叫：

```text
Pre-LN
```

如果是：

```text
先 Attention，再残差相加，再 LayerNorm
```

叫：

```text
Post-LN
```

现在很多 GPT 类模型都更偏向 Pre-LN，因为训练更稳定。

---

## 对照你的 Block

如果你自己的 miniGPT 写的是：

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

那你和 nanoGPT 一样。

如果你写的是：

```python
x = self.ln1(x + self.sa(x))
x = self.ln2(x + self.ffwd(x))
```

那你的版本是 Post-LN。

你现在要记：

```text
nanoGPT 的 Block 是 Pre-LN + residual。
```

---

# 6. `GPTConfig`

源码：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

这是模型配置。

你要逐个理解。

---

## `block_size`

```text
最大上下文长度。
```

比如：

```text
block_size = 1024
```

表示模型最多一次看 1024 个 token。

训练时 `x` 的长度不能超过它：

```python
assert t <= self.config.block_size
```

---

## `vocab_size`

```text
词表大小。
```

默认是：

```python
vocab_size: int = 50304
```

注释里说，GPT-2 原始词表大小是 50257，但这里 padding 到 50304，接近 64 的倍数，为了效率。

这个是工程细节：

```text
有时候为了 GPU 计算效率，会把 vocab size 补齐到某个倍数。
```

---

## `n_layer`

```text
Transformer block 的层数。
```

GPT-2 small 是 12 层。

---

## `n_head`

```text
attention head 数。
```

GPT-2 small 是 12 个 head。

---

## `n_embd`

```text
每个 token 的 hidden size / embedding dimension。
```

GPT-2 small 是 768。

---

## `dropout`

dropout 概率。

训练小模型或大模型时，这个值可以不同。

---

## `bias`

是否在线性层和 LayerNorm 中使用 bias。

注释里说：

```text
True：像 GPT-2 一样使用 bias。
False：可能稍微更好、更快。
```

对你来说，先记：

```text
这个开关控制 Linear 和 LayerNorm 是否带 bias。
```

---

# 7. `GPT.__init__`

这是整个模型主体。

源码结构：

```python
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        ...
```

---

## 7.1 保存 config

```python
assert config.vocab_size is not None
assert config.block_size is not None
self.config = config
```

说明构造 GPT 必须知道：

```text
词表大小
最大上下文长度
```

---

## 7.2 `self.transformer = nn.ModuleDict(...)`

核心代码：

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
```

这个就是 GPT 主体。

逐个看。

---

## `wte`

```python
wte = nn.Embedding(config.vocab_size, config.n_embd)
```

`wte` 是：

```text
word token embedding
```

它把 token id 变成向量。

如果：

```text
idx.shape = (B, T)
```

经过 `wte` 后：

```text
tok_emb.shape = (B, T, n_embd)
```

---

## `wpe`

```python
wpe = nn.Embedding(config.block_size, config.n_embd)
```

`wpe` 是：

```text
word position embedding
```

它把位置编号变成向量。

如果当前序列长度是 `T`：

```python
pos = torch.arange(0, t)
```

那么：

```text
pos.shape = (T,)
pos_emb.shape = (T, n_embd)
```

然后 `tok_emb + pos_emb`，PyTorch 会自动广播成：

```text
(B, T, n_embd)
```

---

## `drop`

```python
drop = nn.Dropout(config.dropout)
```

token embedding 和 position embedding 相加之后做 dropout。

---

## `h`

```python
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
```

这是多个 Transformer block。

比如：

```text
n_layer = 12
```

那就是 12 个 Block。

这对应你的：

```python
self.blocks = nn.Sequential(*[Block(...) for _ in range(n_layer)])
```

nanoGPT 用 `ModuleList`，然后在 forward 里手动循环：

```python
for block in self.transformer.h:
    x = block(x)
```

---

## `ln_f`

```python
ln_f = LayerNorm(config.n_embd, bias=config.bias)
```

这是最后一层 LayerNorm。

GPT 结构通常是：

```text
embedding
↓
blocks
↓
final layer norm
↓
lm_head
```

---

# 7.3 `lm_head`

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

这个层把 hidden state 映射回词表 logits。

如果：

```text
x.shape = (B, T, n_embd)
```

那么：

```text
logits.shape = (B, T, vocab_size)
```

每个位置输出一个词表大小的分数。

---

# 7.4 Weight tying

源码：

```python
self.transformer.wte.weight = self.lm_head.weight
```

这是一个很重要的工程 / 模型技巧：

```text
输入 token embedding 和输出 lm_head 共享同一份权重。
```

叫：

```text
weight tying
```

也就是：

```text
wte.weight 和 lm_head.weight 是同一个参数。
```

为什么可以这样？

输入 embedding 是：

```text
token id → token vector
```

输出 lm_head 是：

```text
hidden vector → 每个 token 的分数
```

它们都和“token 的语义表示”有关，所以可以共享权重。

好处：

```text
1. 减少参数量
2. 有时提升语言模型效果
3. GPT 系列模型常用
```

你自己的 miniGPT 里可能没有做这个。  
这是一个值得你补进去的小工程点。

---

# 7.5 初始化权重

```python
self.apply(self._init_weights)
```

这会递归访问所有子模块，对 Linear 和 Embedding 初始化。

后面函数是：

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

也就是说：

```text
Linear weight: 正态分布，std=0.02
Linear bias: 0
Embedding weight: 正态分布，std=0.02
```

你自己的 toy 版本可能完全使用 PyTorch 默认初始化。  
nanoGPT 显式模仿 GPT-2 的初始化风格。

---

# 7.6 residual projection 的特殊初始化

源码：

```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

这个比较高级，但你现在要知道它在做什么。

`c_proj.weight` 出现在两个地方：

```text
Attention 的输出投影：attn.c_proj
MLP 的输出投影：mlp.c_proj
```

这两个输出都会进入 residual branch：

```python
x = x + attention_output
x = x + mlp_output
```

随着层数变深，残差不断累积，数值可能变大。

所以 nanoGPT 对这些 residual projection 做了更小的初始化：

```text
std = 0.02 / sqrt(2 * n_layer)
```

意思是：

```text
层数越深，每个 residual 输出初始化得越小。
```

你现在不用深究推导，只要记：

```text
这是为了深层 Transformer 训练稳定。
```

---

# 7.7 打印参数量

```python
print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
```

这就是工程项目常见做法：

```text
模型创建后，直接报告参数量。
```

你自己的 miniGPT 也可以加。

---

# 8. `get_num_params`

源码：

```python
def get_num_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params
```

它用来统计参数量。

`non_embedding=True` 时，会减掉 position embedding 的参数量。

注释里解释：token embedding 因为和最终 `lm_head` 共享，所以仍然算进去；position embedding 则可以减掉。

你现在只要知道：

```text
这是一个模型规模统计函数。
```

---

# 9. `forward`

这是你最应该认真看的部分。

源码核心：

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)

    for block in self.transformer.h:
        x = block(x)

    x = self.transformer.ln_f(x)

    if targets is not None:
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
    else:
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss
```

---

## 9.1 输入 `idx`

```python
idx
```

形状是：

```text
(B, T)
```

每个元素是 token id。

例如：

```text
idx =
[
  [10, 25, 93, 102],
  [7, 18, 6, 204]
]
```

---

## 9.2 targets 可有可无

```python
targets=None
```

这很重要。

训练时：

```python
logits, loss = model(x, y)
```

推理时：

```python
logits, loss = model(idx)
```

所以 `forward` 同时服务于：

```text
训练
推理
```

---

## 9.3 不能超过 block_size

```python
assert t <= self.config.block_size
```

如果模型最大上下文长度是 1024，你不能传 2048 个 token 进去。

---

## 9.4 生成 position ids

```python
pos = torch.arange(0, t, dtype=torch.long, device=device)
```

如果 `t=5`：

```text
pos = [0, 1, 2, 3, 4]
```

然后：

```python
pos_emb = self.transformer.wpe(pos)
```

得到每个位置的向量。

---

## 9.5 token embedding + position embedding

```python
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

形状：

```text
idx.shape     = (B, T)
tok_emb.shape = (B, T, C)
pos_emb.shape = (T, C)
x.shape       = (B, T, C)
```

这里的核心就是：

```text
每个 token 的初始表示 =
token embedding + position embedding
```

---

## 9.6 经过所有 Transformer Block

```python
for block in self.transformer.h:
    x = block(x)
```

如果有 12 层，就是重复 12 次：

```text
LayerNorm
CausalSelfAttention
Residual
LayerNorm
MLP
Residual
```

---

## 9.7 最后一层 LayerNorm

```python
x = self.transformer.ln_f(x)
```

得到最终 hidden states。

---

## 9.8 如果是训练：计算所有位置 logits 和 loss

```python
if targets is not None:
    logits = self.lm_head(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1
    )
```

这里要特别理解。

`logits` 原本形状是：

```text
(B, T, vocab_size)
```

`targets` 原本形状是：

```text
(B, T)
```

但是 `F.cross_entropy` 通常希望输入形状类似：

```text
(N, C)
```

其中：

```text
N = 样本数量
C = 类别数量
```

所以它把：

```text
logits:  (B, T, vocab_size)
```

拉平成：

```text
(B*T, vocab_size)
```

把：

```text
targets: (B, T)
```

拉平成：

```text
(B*T)
```

也就是说：

```text
每一个 token 位置都是一个分类任务。
```

目标是预测下一个 token。

---

## `ignore_index=-1`

这个意思是：

```text
如果 target 某些位置是 -1，就不计算这些位置的 loss。
```

这在某些任务里很有用，比如只想对部分 token 计算 loss。

普通预训练里，targets 通常不会有 -1。

---

## 9.9 如果是推理：只算最后一个位置 logits

```python
else:
    logits = self.lm_head(x[:, [-1], :])
    loss = None
```

这是一个很聪明的小优化。

推理生成时，我们只关心最后一个 token 后面接什么。

例如输入：

```text
我 今天 很
```

我们只需要预测：

```text
下一个 token
```

不需要重新输出所有位置的 logits。

所以它只取：

```python
x[:, [-1], :]
```

注意这里写的是 `[-1]`，不是 `-1`。

区别是：

```python
x[:, -1, :].shape
```

会变成：

```text
(B, C)
```

而：

```python
x[:, [-1], :].shape
```

保持：

```text
(B, 1, C)
```

这样输出 logits 仍然是三维：

```text
(B, 1, vocab_size)
```

这个是工程上的小细节。

---

# 10. `crop_block_size`

源码：

```python
def crop_block_size(self, block_size):
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```

这个函数的作用是：

```text
把模型支持的最大上下文长度裁短。
```

比如：

```text
原来 GPT-2 block_size = 1024
现在我只想用 256
```

那就裁剪 position embedding：

```python
self.transformer.wpe.weight[:block_size]
```

如果 attention 里有 causal mask，也裁剪：

```python
block.attn.bias[:,:,:block_size,:block_size]
```

这个函数通常用于：

```text
加载 GPT-2 预训练模型后，用更短上下文微调。
```

你自己的 miniGPT 大概率没有这个功能。

这就是工程化场景：

```text
模型结构可能需要根据实际任务做 surgery。
```

---

# 11. `from_pretrained`

这个函数比较长。第一遍不需要完全吃透，但要知道它干什么。

作用：

```text
从 Hugging Face 的 GPT-2 checkpoint 加载权重到 nanoGPT 的 GPT 结构中。
```

支持：

```python
{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
```

---

## 11.1 根据模型类型确定结构

源码里有：

```python
config_args = {
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
}[model_type]
```

也就是：

|模型|层数|head 数|embedding|
|---|--:|--:|--:|
|GPT-2|12|12|768|
|GPT-2 medium|24|16|1024|
|GPT-2 large|36|20|1280|
|GPT-2 XL|48|25|1600|

---

## 11.2 GPT-2 固定参数

源码强制设置：

```python
config_args['vocab_size'] = 50257
config_args['block_size'] = 1024
config_args['bias'] = True
```

因为 GPT-2 checkpoint 本身就是这些设置。

这说明：

```text
加载预训练权重时，模型结构必须和 checkpoint 对得上。
```

否则参数 shape 对不上。

---

## 11.3 创建 nanoGPT 模型

```python
config = GPTConfig(**config_args)
model = GPT(config)
```

先创建一个 nanoGPT 模型。

---

## 11.4 加载 Hugging Face GPT-2

```python
from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained(model_type)
sd_hf = model_hf.state_dict()
```

这里调用 Hugging Face 下载 / 加载 GPT-2 权重。

---

## 11.5 复制参数

核心逻辑是：

```text
把 Hugging Face GPT-2 的参数复制到 nanoGPT 的参数里。
```

但是有些权重要转置：

```python
transposed = [
    'attn.c_attn.weight',
    'attn.c_proj.weight',
    'mlp.c_fc.weight',
    'mlp.c_proj.weight'
]
```

为什么？

源码注释说，OpenAI checkpoint 使用的是一种叫 `Conv1D` 的模块，而 nanoGPT 用的是普通 `Linear`，所以导入时需要转置。

你现在不需要深挖，只要知道：

```text
同一个数学操作，不同框架里参数矩阵存储方向可能不同。
```

所以加载权重时需要做 shape 对齐。

---

## 11.6 你现在要怎么处理这个函数？

第一遍只需要记：

```text
from_pretrained 不是模型 forward 的核心。
它是为了加载 GPT-2 预训练权重。
```

你现在可以先略读。等读 `sample.py` 和 `finetune` 时再回来。

---

# 12. `configure_optimizers`

这个函数非常重要，因为它展示了训练工程里 optimizer 怎么配置。

源码核心：

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args
    )
```

你自己的版本可能是：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
```

nanoGPT 更细。

---

## 12.1 参数分成两组

第一组：

```python
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
```

这些参数使用 weight decay。

一般包括：

```text
Linear weight
Embedding weight
```

第二组：

```python
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
```

这些参数不使用 weight decay。

一般包括：

```text
bias
LayerNorm weight
```

---

## 12.2 为什么要这样分？

简单理解：

```text
矩阵权重可以做 weight decay；
bias 和 LayerNorm 参数通常不做 weight decay。
```

这是训练 Transformer 时常见的优化器配置。

你可以记成：

```text
大矩阵参数：decay
小向量参数：no decay
```

---

## 12.3 fused AdamW

源码：

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
```

意思是：

```text
如果 PyTorch 支持 fused AdamW，并且当前是 CUDA，就使用更快版本。
```

这又是一个工程性能优化点。

---

# 13. `estimate_mfu`

源码：

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
```

这个函数用于估计：

```text
模型在 A100 GPU 上的理论算力利用率。
```

MFU 大概可以理解为：

```text
实际训练吞吐 / 理论最高吞吐
```

这个对你第一遍学习不是重点。

可以先跳过。

你只需要知道：

```text
这是性能分析函数，不影响模型结构。
```

---

# 14. `generate`

这是你已经写过的生成函数的工程版。

源码核心：

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

我们细看。

---

## 14.1 `@torch.no_grad()`

表示生成时不计算梯度。

因为推理不需要反向传播。

好处：

```text
省显存
更快
```

---

## 14.2 输入 idx

```python
idx
```

形状：

```text
(B, T)
```

是 prompt 的 token ids。

---

## 14.3 循环生成 token

```python
for _ in range(max_new_tokens):
```

每次生成一个新 token，一共生成 `max_new_tokens` 个。

---

## 14.4 如果上下文太长，就裁剪

```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
```

假设模型最大上下文是：

```text
block_size = 1024
```

如果当前已经生成了 1500 个 token，模型不能一次看 1500 个，只能看最后 1024 个：

```text
取最后 block_size 个 token 作为上下文。
```

这就是滑动窗口式生成。

---

## 14.5 forward 得到 logits

```python
logits, _ = self(idx_cond)
```

因为没有 targets，所以 forward 走的是推理分支：

```python
logits = self.lm_head(x[:, [-1], :])
```

也就是只输出最后一个位置的 logits。

---

## 14.6 temperature

```python
logits = logits[:, -1, :] / temperature
```

temperature 控制生成随机性。

```text
temperature < 1：
分布更尖锐，更保守。

temperature = 1：
不改变分布。

temperature > 1：
分布更平，更随机。
```

例如：

```text
temperature = 0.8
```

通常更稳。

```text
temperature = 1.2
```

通常更发散。

---

## 14.7 top_k

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```

意思是：

```text
只保留概率最高的 k 个 token，其余 token 设为 -inf。
```

比如：

```text
top_k = 50
```

那么每一步只从最可能的 50 个 token 里采样。

这样可以避免模型采到极低概率的奇怪 token。

---

## 14.8 softmax 得到概率

```python
probs = F.softmax(logits, dim=-1)
```

把 logits 转成概率分布。

---

## 14.9 multinomial 采样

```python
idx_next = torch.multinomial(probs, num_samples=1)
```

这和你之前 greedy generation 不同。

你的 greedy 可能是：

```python
idx_next = torch.argmax(probs, dim=-1)
```

nanoGPT 是：

```text
根据概率分布随机抽样。
```

所以它不是永远选最大概率 token。

---

## 14.10 拼接新 token

```python
idx = torch.cat((idx, idx_next), dim=1)
```

把新 token 接到序列后面，然后进入下一轮。

---

# 15. generate 和你的 greedy generation 对比

|项目|你的 greedy|nanoGPT generate|
|---|---|---|
|下一 token|选最大概率|从概率分布采样|
|随机性|没有|有|
|temperature|通常没有|有|
|top_k|通常没有|有|
|上下文超长处理|可能没有|会裁剪到 block_size|
|梯度|可能没显式关闭|`@torch.no_grad()`|

这个是你可以马上吸收的工程升级点。

---

# 16. 整个 forward 的数据流

你可以把 `GPT.forward()` 画成这样：

```text
idx: (B, T)
↓
wte(idx)
token embedding: (B, T, C)
↓
wpe(pos)
position embedding: (T, C)
↓
tok_emb + pos_emb
x: (B, T, C)
↓
dropout
↓
Block 1
↓
Block 2
↓
...
↓
Block n
↓
final LayerNorm
↓
lm_head
↓
logits: (B, T, vocab_size)
↓
cross entropy loss
```

如果是推理：

```text
idx
↓
GPT body
↓
只取最后一个位置
↓
lm_head
↓
最后一个 token 的 logits
↓
temperature / top_k / softmax / sample
↓
拼接新 token
```

---

# 17. 你应该重点掌握的代码段

第一遍读 `model.py`，你最应该吃透这 6 个地方：

## 第一，QKV 合并

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

这对应：

```text
工程版 QKV 计算。
```

---

## 第二，多头 reshape

```python
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

这对应：

```text
(B, T, C) → (B, n_head, T, head_size)
```

---

## 第三，causal attention

```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
```

这对应：

```text
不能看未来 token。
```

---

## 第四，Pre-LN Block

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

这对应：

```text
LayerNorm → 子层 → 残差。
```

---

## 第五，GPT forward

```python
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)
```

这对应：

```text
你自己的 MiniGPT 主体。
```

---

## 第六，generate

```python
logits = logits[:, -1, :] / temperature
...
idx_next = torch.multinomial(probs, num_samples=1)
```

这对应：

```text
采样生成，不是 greedy。
```

---

# 18. 这个文件里你第一遍可以暂时跳过的部分

为了不被细节淹没，第一遍可以跳过：

```text
from_pretrained 里具体每个 key 怎么对齐
estimate_mfu 的 FLOPs 公式
fused AdamW 的 inspect 细节
torch.compile warning
多 GPU 相关内容，因为 model.py 里基本没有
```

但是不能跳过：

```text
LayerNorm
CausalSelfAttention
MLP
Block
GPTConfig
GPT.forward
GPT.generate
configure_optimizers 的大意
```

---

# 19. 和你自己的 miniGPT 的总对照

|你自己的 miniGPT|nanoGPT 中对应位置|nanoGPT 多了什么|
|---|---|---|
|token embedding|`transformer.wte`|weight tying|
|position embedding|`transformer.wpe`|可裁剪 block size|
|masked self-attention|`CausalSelfAttention`|QKV 合并、Flash Attention|
|Transformer block|`Block`|Pre-LN、可配置 bias|
|FFN|`MLP`|GPT-2 风格命名和初始化|
|MiniGPT|`GPT`|参数统计、权重初始化、加载 GPT-2|
|training loss|`forward(..., targets)`|`ignore_index=-1`|
|greedy generate|`generate()`|temperature、top_k、multinomial|
|optimizer|`configure_optimizers()`|weight decay 参数分组、fused AdamW|

---

# 20. 你读完 `model.py` 应该形成的核心认识

这份 `model.py` 并没有引入一个你完全不懂的新模型。

它本质还是：

```text
token embedding
+ position embedding
+ 多层 Transformer decoder block
+ language modeling head
```

但它比你的 toy miniGPT 多了这些工程增强：

```text
1. 配置对象 GPTConfig
2. QKV 合并计算
3. Flash Attention 支持
4. 自定义 LayerNorm bias 开关
5. Pre-LN Transformer block
6. GPT-2 风格权重初始化
7. residual projection 特殊初始化
8. token embedding 和 lm_head 权重共享
9. 训练和推理共用 forward
10. 推理时只计算最后一个位置 logits
11. 支持加载 GPT-2 预训练权重
12. optimizer 参数分组
13. 支持 top_k / temperature 采样
```

你现在最应该记住一句话：

```text
nanoGPT 的 model.py = 你手搓 MiniGPT 的工程增强版。
```

下一步最适合继续看 `train.py`，因为你已经知道模型怎么定义了，接下来就该看：

```text
这个模型是怎么被训练起来的。
```