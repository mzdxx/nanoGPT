"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math     # 主要用于attention里的缩放
import inspect          # 在后面用于判断当前Pytroch的AdamW是否支持fused=name
from dataclasses import dataclass       # 用于定义GPTConfig，比普通class简洁，适合保存配置

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):     # 重复Pytorch的nn.LayerNorm,但是允许LayerNorm不使用bias
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
# 层归一化，对embedding维度做归一化

class CausalSelfAttention(nn.Module):

    def __init__(self, config):     
        super().__init__()
        assert config.n_embd % config.n_head == 0       # 多头自注意力，embedding维度必须能被head数整除
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias) # QKV合并，使用一次Linear输出三份结果，然后在forward里面再切开
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)     # 多头attention拼接回来后，经过一个线性层
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)      # 在forward中作用在用在attention权重上面
        self.resid_dropout = nn.Dropout(config.dropout)     # 在attention输出投影之后
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')       # 检查对象有没有某种方法，这里就是检查这个模块里有没有这个函数
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence，没scaled_dot_product_attention就注册一个mask，下三角矩阵，让当前位置不能偷看未来token
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)     # 把合并的QKV分离
        # reshape成多头模式，并交换第1，2维的位置，这样保持batch,n_head,length,n_headembedding,也就是批次和头数在前
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # 因果自注意力
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention    手动实现attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 交换1，2维的顺序，多头拼接回去
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        


        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):       # Multi-Layer Perception  多层感知机

    def __init__(self, config):     # 先升维度，再降低维度，中间还有非线性层
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

class Block(nn.Module):     # 就是一个快或者说模块，在Transformer里特指一个基本的计算单元，内部结构一般是注意力层+前馈网络

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))         # 前置归一化的Transformer
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024      # 最大上下文长度
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency     词表大小
    n_layer: int = 12       # Transformer block的层数
    n_head: int = 12           # 头的数量
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # 词嵌入，把tokenid变成向量
            wpe = nn.Embedding(config.block_size, config.n_embd),   # 位置嵌入
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 多个transformer block块
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # language model head 把这个层hidden state映射回词表logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate


        # 一个权重约束，让输入token embedding 和输出lm_head共享同一份权重
        # 好处是减少参数量，有时提升语言模型效果，GPT系列模型常用
        # nn.Linear和nn.Embedding分别是一个类，这两个类内部都有一个属性weight

        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying


        # init all weights
        self.apply(self._init_weights)          # apply是继承来的方法来自nn.Module,然后把这个私有方法本身作为参数传递给apply,apply(fn)会递归遍历模型的所有子模块，对每个子模块调用fn(module),这里就是把模型里的每一层都交给_init_weights函数处理进行初始化参数
        # 按照GPT-2论文的方法，对残差投影层做特殊缩放初始化
        for pn, p in self.named_parameters():           # self.named_parameters()是nn.Module的犯法，返回一个迭代器，每次迭代产出(参数名，参数Tensor)的元组
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

                # torch.nn.init是一个模块，normal_是原地正态分布初始化

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        返回模型中的参数数量，默认只计算不包括各类嵌入层的参数，也要把位置编码的参数减掉，按理说
        词嵌入参数也应该一起剪掉，但是因为词嵌入有参数共享，所以词嵌入不能删除，要算进总参数里
        """
        n_params = sum(p.numel() for p in self.parameters())
        # self.parameters()是nn.Module的方法，返回一个生成器，遍历模型的所有可训练参数
        # p.numel()number of elements元素个数是tensor的方法，返回这个Tensor里一共有多少个数字
        # sum(表达式 for 变量 in 可迭代对象)，括号里没有方括号就是一个生成器，逐个产出值给 sum 累加。

        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()         # 把位置编码的参数减去
        return n_params

    def _init_weights(self, module):        # 这是内部使用的方法，外部不应该调用
        if isinstance(module, nn.Linear):           # 返回一个对象是否是类或子类的实例
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)        # 原地正态分布初始化
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)           # 原地填零初始化
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):       # 前向传播，这里targets是可选参数，训练时传递目标序列，推理时不传递
        device = idx.device
        b, t = idx.size()           # batch_len seq_len
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t),array range用来生成位置编码的索引，这里还规定数据类型为长整型

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)        # 这里的加法有广播机制
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)        # 最后的LayerNorm,用来稳定输出

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)        # 把最后一层隐藏层状态映射到词表大小
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # 交叉熵损失函数，这里.view(-1, logits.size(-1))把logits拉成(b*t, vocab_size)，
            # targets拉成(b*t,)，拉平成二维
            # 忽略target，也就是标签为-1的位置（padding）
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # [-1]最后一个位置，用列表包起来，保持维度
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # 如果有必要，通过模型修改来减少块的上下文窗口的大小，例如我们可能加载了GPT2预训练权重，block_size=1024,但是希望给一些更小，更简单的模型使用更小的块大小
        assert block_size <= self.config.block_size
        self.config.block_size = block_size         # 更新config.block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

        #self.transformer.wpe.weight，位置嵌入表，形状 (1024, n_embd)（假设原始 block_size=1024）
        # 这里切片只取前block_size行
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):     # 涉及到上下文窗口的，也就是和自注意力相关
                # 这里的bias是因果注意力里的下三角mask
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]  # 前两维全取，后两个维度裁切到新长度

    @classmethod        # 类方法修饰符，model = GPT.from_pretrained('gpt2')不需要实例化，直接调用类名
    def from_pretrained(cls, model_type, override_args=None):
        # 从 Hugging Face 的 GPT-2 checkpoint 加载权重到 nanoGPT 的 GPT 结构中。
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict，如果overide_args覆盖参数
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()         # 返回一个有序字典
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param过滤掉以.attn.bias结尾的，这是mask不是训练参数

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # 配置优化器，把参数分成两组，该做权重衰减的和不该做的(bias,LayerNorm)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters从所有候选参数开始
        param_dict = {pn: p for pn, p in self.named_parameters()}       # 字典{参数名，参数}
        # filter out those that do not require grad筛选出不需要梯度的
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # param_dict.items()字典方法，返回键值对，还有一个if过滤条件
        # 创建优化器参数分组，所有2维的参数会做权重衰减，其余参数不做，也就是说所有矩阵乘法，嵌入层的权重张量会衰减，所有偏置项和层归一化参数不衰减
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]       # 告诉优化器dacay_params正常衰减，nodecay_params不做衰减
        # 分别统计两组各有多少个参数
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # 创建AdamW优化器，可能的话使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters

        # torch.optim.AdamW是AdamW优化器类，
        # inspect.signature(torch.optim.AdamW)获取AdamW的函数签名，也就是__init__接受什么参数
        # inspect.signature(torch.optim.AdamW).parameters返回参数字典，键是参数名，值是 Parameter 对象

        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # **extra_args字典解包
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # 估计模型的MFU，也就是Model FLOPs Utilization 模型浮点运算利用率，看看用了GPU峰值算力的多少
    # 这里的GPU是A100
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # 按照 PaLM 论文的公式，估算一次前向+反向传播需要的浮点运算次数
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()    # 装饰器，在这个函数执行期间不计算梯度，省显存

    # 给定起始token(idx) 自回归生成max_new_token个新的token
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        输入一段已经有的文本索引序列，(b,t),让模型自动往后继续生成max_new_tokens个新内容，
        每次生成一个词，就继续把这个新词喂回模型，模型是评估模式
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # 
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]           # 一个三元表达式，如果序列太长就只保留最后block_size个token
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)      # 调用模型的forward，通过__call__触发
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # 只取序列最后一个位置的logits，这里没加方括号，结果会降维成(batch_size,vocab_size)



            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 返回(value,indices)
                # 注意这里是按照最后一个维度，取最大的k个值
                # v,_ 表示只要值不要索引，v的形状是(batch,top_k)



                # torch.topk(logits, k)返回(values,indices) 只要值不要索引


                logits[logits < v[:, [-1]]] = -float('Inf')
                # v[:,[-1]]取每个batch的第k大的值，也就是阈值，保持维度，形状 (batch, 1) 而不是 (batch,)
                # logits[logits < v[:, [-1]]]这里[]是一个布尔索引，也就是选出所有小于阈值的logits，并把这些设为负无穷
                # 这里的logits是(batchsize,vocab_size)


            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)       # 沿着最后一个维度做softmax,把logits变成总和为一的概率，前面的负无穷softmax后就变成了0
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)      # multinomial 多项采样分布，
            # 从概率分布probs,采一个样本，返回的是索引tokenID,形状(batch,1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # idx 形状 (b, t)，idx_next 形状 (b, 1)

            # 这里的top_k采样写的有点复杂后续继续看

        return idx
