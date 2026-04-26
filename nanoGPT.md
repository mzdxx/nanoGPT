这是用于训练 / 微调中等规模 GPT 模型最简单、最高效的代码仓库。它是 [minGPT](https://github.com/karpathy/minGPT) 的重构版本，更侧重实际使用价值而非教学属性。
**代码**
代码本身简洁易读：`train.py` 是约 300 行的基础训练循环代码，`model.py` 是约 300 行的 GPT 模型定义代码，还可选择性地加载 OpenAI 发布的 GPT-2 预训练权重。核心代码仅此而已。

依赖项安装
pip install torch numpy transformers datasets tiktoken wandb tqdm

nanoGPT的工作流程

```
prepare.py
负责数据预处理

train.py + config
负责训练

sample.py
负责加载模型并生成文本
```


看model.py

```
self.transformer = nn.ModuleDict(dict(

            wte = nn.Embedding(config.vocab_size, config.n_embd),   # 词嵌入，把tokenid变成向量

            wpe = nn.Embedding(config.block_size, config.n_embd),   # 位置嵌入

            drop = nn.Dropout(config.dropout),

            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 多个transformer block块

            ln_f = LayerNorm(config.n_embd, bias=config.bias),

        ))
```


```
# 一个权重约束，让输入token embedding 和输出lm_head共享同一份权重

        # 好处是减少参数量，有时提升语言模型效果，GPT系列模型常用

        # nn.Linear和nn.Embedding分别是一个类，这两个类内部都有一个属性weight

  

        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

  
  

        # init all weights

        self.apply(self._init_weights)          # apply是继承来的方法来自nn.Module,然后把这个私有方法本身作为参数传递给apply
```


**权重衰减**


对！**100% 完全正确！** 我给你用**最稳、最准、最不绕弯**的话确认一遍👇 # 1. 权重 = 参数 在深度学习里： **权重（weight）= 模型里的参数（parameter）** 这俩词**基本就是同一个东西**。 --- # 2. 你理解的更新公式完全正确 **正常更新（所有参数都用）：** ``` 新参数 = 旧参数 - 学习率 × 梯度 ``` **有权重衰减的更新（只给矩阵参数用）：** ``` 新参数 = 旧参数 - 学习率×梯度 - 学习率×衰减率×旧参数 ``` 就是**多减了一小点**，让参数变得更小一点。 --- # 3. 回到代码那句，它到底在干嘛？ 代码说： ``` # 二维参数（矩阵）做权重衰减 # 一维参数（偏置、层归一化）不做 ``` 翻译成**你能完全听懂**的话： ### 训练时： - **矩阵类参数**（比如 W 权重） 更新规则 = **正常更新 + 额外拉小一点（衰减）** - **向量类参数**（比如 bias、layernorm） 更新规则 = **只做正常更新，不拉小** --- # 4. 终极极简总结（你已经完全懂了） - **权重 = 参数** - **权重衰减 = 给一部分参数多加一个“变小”的更新项** - **代码就是在分两类更新：** - 矩阵 → 多减一点（衰减） - 向量 → 正常减（不衰减） 你现在**完全理解**这段代码在干嘛了！ 要不要我再给你画一张**超简单小表格**，一眼看清哪些参数衰减、哪些不衰减？
