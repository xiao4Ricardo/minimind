# MiniMind 源码手册

> 从一个 token 走完整条训练链路
>
> 骨架 hidden 768 · 8 层 · vocab 6400 ｜ 注意力 GQA 8Q/4KV · head_dim 96
> 参数量 dense 63.91M / MoE 198.42M-A63.94M ｜ 实测环境 单卡 RTX 5060 8GB · 86 GPU 小时

这份手册把 MiniMind 的每一处设计拆到源码行，再拆到它背后的数学。所有参数量、张量形状、超参默认值都取自本仓库 `model/model_minimind.py` 等文件的实际代码，不是从论文或博客转述的通用知识。读完你应该能回答的不只是「RoPE 是什么」，而是「这一行 `torch.cat([cos, cos])` 为什么这么写」。

第五章的实测数据全部来自本仓库在单张 RTX 5060 8GB 上的真实训练，六个阶段合计 86.4 GPU 小时，零崩溃零重拉。

---

## 目录

**第一章 · 全局架构**
- [1.1 六阶段全生命周期](#11-六阶段全生命周期)
- [1.2 目录结构与模块职责](#12-目录结构与模块职责)
- [1.3 一个 token 的完整旅程](#13-一个-token-的完整旅程)

**第二章 · 模型结构**
- [2.1 配置速查与参数量核算](#21-配置速查与参数量核算)
- [2.2 RMSNorm vs LayerNorm](#22-rmsnorm-vs-layernorm)
- [2.3 RoPE 与 YaRN 外推](#23-rope-旋转位置编码)
- [2.4 GQA · QK-Norm · KV Cache](#24-gqaqk-normkv-cache)
- [2.5 SwiGLU](#25-swiglu-前馈网络)
- [2.6 MoE 与负载均衡](#26-moe-与负载均衡)
- [2.7 权重绑定](#27-权重绑定-tie_word_embeddings)

**第三章 · 训练机制**
- [3.1 自回归 Loss 与因果掩码](#31-自回归-loss-与因果掩码)
- [3.2 Pretrain 与 SFT 的唯一本质差别](#32-pretrain-与-sft-的唯一本质差别)
- [3.3 学习率调度](#33-学习率调度)
- [3.4 混合精度 · 累积 · 裁剪](#34-混合精度--梯度累积--梯度裁剪)
- [3.5 DPO](#35-dpo-直接偏好优化)
- [3.6 LoRA](#36-lora-低秩适配)
- [3.7 GRPO / CISPO / PPO](#37-grpo--cispo--ppo-与-agentic-rl)

**第四章 · 数据管线**
- [4.1 Tokenizer 与 BPE](#41-tokenizer-与-byte-level-bpe)
- [4.2 ChatML 模板](#42-chatml-模板)
- [4.3 四种 Dataset 对比](#43-四种-dataset-对比)

**第五章 · 实测参照**
- [5.1 资源与耗时参照表](#51-资源与耗时参照表)
- [5.2 收敛参照](#52-收敛参照loss-降到多少算好)
- [5.3 效果对比与消融](#53-效果对比与消融)
- [5.4 8GB 显存工程](#54-8gb-显存工程)

**第六章 · 面试题**
- [20 题与回答模板](#第六章--面试高频题与回答模板)
- [附：一页速查](#附一页速查)

---

# 第一章 · 全局架构与流程

## 1.1 六阶段全生命周期

MiniMind 不是「拿一个预训练模型微调」，而是**从随机初始化开始、把整条链路走完**。每个阶段的输入是上一阶段的权重，输出是一个新的 `.pth`。理解这条链最重要的一点是：**每一步换的是「学什么信号」，模型结构自始至终没变。**

| 阶段 | 脚本 | 做什么 | 输入 → 输出 |
| --- | --- | --- | --- |
| **0 · Tokenizer** | `train_tokenizer.py` | 把文字变成整数。Byte-level BPE，词表 6400。一旦定死，后面所有权重都绑在这个词表上，换词表等于全部重训 | → `model/tokenizer.json` |
| **1 · Pretrain** | `train_pretrain.py` | 学语言本身。纯文本自回归，**每一个 token 都算 loss**。学到「中文长什么样」，但完全不会对话 —— 你问它问题，它接着往下写文章 | `pretrain_t2t_mini.jsonl` → `pretrain_768.pth` |
| **2 · SFT** | `train_full_sft.py` | 学对话格式。同样是交叉熵，但**只对 assistant 段落算 loss**，prompt 标 `-100`。这是与 Pretrain 唯一的本质差别 | `sft_t2t_mini.jsonl` → `full_sft_768.pth` |
| **3 · 偏好对齐** | `train_dpo.py` | 学「人更喜欢哪个」。chosen/rejected 成对数据，不需要奖励模型、不需要采样 | `dpo.jsonl` → `dpo_768.pth` |
| **4 · 策略优化** | `train_grpo.py` `train_ppo.py` | 学「怎么拿高分」。模型自己采样、奖励模型打分、按优势更新。GRPO/CISPO 用组内相对优势省掉 critic | `rlaif.jsonl` + 奖励模型 → `grpo_768.pth` |
| **5 · 轻量化/部署** | `train_lora.py` `eval_llm.py` | LoRA 只训 0.39M 参数（占 0.62%）；保存时统一 `.half()` 转 fp16，所以 63.91M 的权重文件只有 131 MB | → `lora_*.pth` |

> **面试常问：为什么 SFT 之后还要 DPO / RL？**
>
> SFT 是**模仿**：它只能告诉模型「这个回答是对的」，永远给不出「这个比那个好」，更给不出「这个是错的」。一旦标注数据里存在风格不一致或质量参差，SFT 会把好坏一起学进去。偏好对齐引入的是**相对信号**（A 优于 B）和**负向信号**（不要这样答），这是交叉熵表达不了的。

## 1.2 目录结构与模块职责

| 路径 | 职责 | 关键内容 |
| --- | --- | --- |
| `model/model_minimind.py` | 模型全部定义 | Config、RMSNorm、RoPE、Attention、FeedForward、MOEFeedForward、Block、CausalLM、自实现 `generate` |
| `model/model_lora.py` | LoRA 注入 | `apply_lora` 用 monkey-patch 改写 `forward`，不改模型定义 |
| `model/tokenizer.json` | 词表 | 6400 词，ChatML 特殊 token |
| `dataset/lm_dataset.py` | 四种数据集 | Pretrain / SFT / DPO / RLAIF，**标签与 mask 的差异全在这里** |
| `trainer/train_*.py` | 各阶段训练循环 | 每个文件自带 `train_epoch` 与 `argparse`，彼此独立、互不继承 |
| `trainer/trainer_utils.py` | 公共工具 | `get_lr`、`setup_seed`、`lm_checkpoint`、`SkipBatchSampler`、`init_model` |
| `trainer/rollout_engine.py` | RL 采样引擎 | 把「策略推理」与「训练」解耦，可插拔换 SGLang |

> **读码顺序建议**：先看 `model_minimind.py`（一个文件读懂整个模型），再看 `lm_dataset.py`（读懂标签怎么造），最后随便挑一个 `train_*.py`（训练循环都长一个样）。**不要从 `train_*.py` 开始读** —— 它们是最容易懂也最没信息量的部分。

## 1.3 一个 token 的完整旅程

把 batch=1、seq_len=512 的一次前向拆开。**面试时能把形状说对，比背概念有说服力得多。**

| 步骤 | 算子 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| 输入 | `input_ids` | `[1, 512]` | 整数 token id，范围 0–6399 |
| 嵌入 | `embed_tokens` | `[1, 512, 768]` | 查表；与 lm_head 共享权重 |
| × 8 层 | `input_layernorm` | `[1, 512, 768]` | RMSNorm，**Pre-Norm** 位置 |
| | `q_proj / k_proj / v_proj` | `[1,512,768]` / `[1,512,384]` ×2 | Q 8 头、KV 各 4 头 → GQA |
| | `q_norm / k_norm` | `[1,512,8,96]` / `[1,512,4,96]` | **QK-Norm**，在 RoPE 之前 |
| | `apply_rotary_pos_emb` | 同上 | 位置信息在此注入 |
| | `repeat_kv(n_rep=2)` | `[1,512,8,96]` | KV 头复制 2 份对齐 Q 头 |
| | SDPA / 朴素注意力 | `[1, 8, 512, 96]` | 因果掩码；朴素路径会实体化 `[1,8,512,512]` |
| | `o_proj` + 残差 | `[1, 512, 768]` | 再过 post_attention_layernorm → SwiGLU → 残差 |
| 输出 | `norm → lm_head` | `[1, 512, 6400]` | 每个位置对全词表的 logits |
| **损失** | `shift + cross_entropy` | 标量 | logits 掐掉最后一位、labels 掐掉第一位 |

> **显存直觉**：最后那步 `[1, 512, 6400]` 看着不大，但训练时 batch=16、seq=768 就是 `16×768×6400×4 B ≈ 300 MB`（fp32），反向还要留一份。**词表维度的 logits 往往是小模型训练里最大的单块激活**，这也是 MiniMind 把词表压到 6400 的直接收益。

---

# 第二章 · 模型结构逐件拆解

## 2.1 配置速查与参数量核算

| 配置项 | 值 | 含义 / 为什么是这个值 |
| --- | ---: | --- |
| `hidden_size` | 768 | 模型宽度 |
| `num_hidden_layers` | 8 | 深度。浅网络训练快，768 的宽度又不至于模式崩溃 |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数 → `n_rep = 8/4 = 2`，标准 GQA |
| `head_dim` | 96 | `768 / 8` |
| `vocab_size` | 6400 | 极小词表。省 embedding 与 logits 显存，代价是同样文本 token 数更多 |
| `intermediate_size` | 2432 | `ceil(768 × π / 64) × 64` —— 用 π 取约 3.17 倍扩张比再对齐到 64 |
| `rope_theta` | 1e6 | 比常见的 1e4 大 100 倍，低频维度周期更长，利于长文本 |
| `rms_norm_eps` | 1e-6 | 数值稳定项 |
| `tie_word_embeddings` | True | 输入嵌入与输出投影共享权重，省 4.92M 参数 |
| `num_experts / per_tok` | 4 / 1 | MoE 时生效：4 专家、top-1 路由 |

### 参数量是怎么算出来的（可手工验算）

面试官很喜欢问「你这 64M 是怎么来的」。逐项拆开如下，三个数字都与训练日志打印的 `Model Params` 完全一致：

```
每层 = 注意力 1,769,664 + MLP 5,603,328 + 两个 RMSNorm 1,536 = 7,374,528
  ├ 注意力 = q 768×768 + k 768×384 + v 768×384 + o 768×768 + qk_norm 192
  └ MLP    = gate 768×2432 + up 768×2432 + down 2432×768

dense 总计 = 8 × 7,374,528 + embed 4,915,200 + final_norm 768 = 63,912,192 ≈ 63.91M

MoE 每层 = 1,769,664 + [gate 3,072 + 4 × 5,603,328] + 1,536 = 24,187,584
MoE 总计 = 8 × 24,187,584 + 4,915,200 + 768 = 198,416,640 ≈ 198.42M
MoE 激活 = top-1 只走 1 个专家 → 8 × 7,377,600 + 4,915,200 + 768 = 63,936,768 ≈ 63.94M
```

注意 `embed_tokens` **只计一次** —— 因为 `tie_word_embeddings=True`，`lm_head` 与它是同一张权重。这也是日志里 MoE 打印成 `198.42M-A63.94M` 的由来：总参 198M，但每个 token 实际只激活 63.94M，**与 dense 的 63.91M 几乎相同 —— 这正是 MoE 对比实验成立的前提。**

## 2.2 RMSNorm vs LayerNorm

```python
# model/model_minimind.py · class RMSNorm · L47–56
def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
    #                              ^^^^^^^^^     ^^^^^^^^^^
```

```
LayerNorm:  y = γ · (x − μ) / √(σ² + ε) + β     （要算均值、方差，还有偏置 β）
RMSNorm:    y = γ · x / √(mean(x²) + ε)         （不减均值、无偏置）
```

**直觉类比**：LayerNorm 是「先把这排数移到以 0 为中心，再缩放到标准长度」；RMSNorm 省掉了移动那一步，**只做缩放**。实践发现 Transformer 里减均值这一步收益很小，去掉后少一次归约、少一组偏置参数，速度更快且效果几乎不变。

> **容易被追问的细节**：`x.float()` 与 `.type_as(x)` 这一对不是多余的。混合精度训练时 `x` 是 bf16，而 `x.pow(2).mean()` 在 bf16 下**极易溢出或损失精度**（768 个数的平方和）。所以先升到 fp32 算归一化，再降回原精度。**「归一化层内部保持 fp32」是所有主流实现的共识做法**，被问到混合精度时这是一个很好的加分点。

> **Pre-Norm 还是 Post-Norm**：看 `MiniMindBlock.forward`：`hidden + self_attn(input_layernorm(hidden))` —— 归一化在**子层之前**，残差是干净的恒等路径，属于 **Pre-Norm**。好处是梯度能沿残差直通、深层也不易发散，代价是最终表示的尺度会随层数累加，所以最后额外加了一个 `self.norm` 收口。

## 2.3 RoPE 旋转位置编码

绝对位置编码（把位置 embedding 加到输入上）有个根本问题：模型学到的是「第 5 个位置」这种绝对概念，换个长度就失效。RoPE 的思路完全不同 —— **不加任何东西，而是把 Q 和 K 向量按位置「转一个角度」。**

```
把 head_dim 的 96 维两两配对成 48 个二维平面，第 i 个平面的旋转角为：

    θ_i = pos / base^(2i/d)          base = rope_theta = 1e6

对每个平面做二维旋转：
    [q'_2i  ]   [cos θ  −sin θ] [q_2i  ]
    [q'_2i+1] = [sin θ   cos θ] [q_2i+1]

关键性质（RoPE 的全部意义所在）：
    ⟨R(m)·q , R(n)·k⟩ = f(q, k, m − n)
```

**直觉类比**：想象每个维度对是一个时钟指针，位置越靠后转得越多。两个 token 做注意力时算的是两根指针的**夹角** —— 而夹角只取决于「差了几个位置」，跟它们各自在第几位无关。**这就是「绝对方式编码、相对方式生效」。**

```python
# model/model_minimind.py · precompute_freqs_cis / apply_rotary_pos_emb · L58–86
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
freqs = torch.outer(torch.arange(end), freqs).float()                  # [seq, 48]
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)    # [seq, 96]

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)
```

> **为什么是 cat 而不是 interleave**
>
> 数学上配对的是 `(x₀,x₁), (x₂,x₃)…`，但代码里 `rotate_half` 配的是 `(x₀, x₄₈), (x₁, x₄₉)…` —— **前半段与后半段配对**。这就是 `cat([cos, cos])` 而不是 `repeat_interleave` 的原因。两种配法在数学上等价（只是维度的一个置换），但**切片比交错快得多**，所以 GPT-NeoX / LLaMA 系全用这一种。**换实现时若两边配法不一致，权重就废了** —— 这是移植 RoPE 最常见的踩坑点。

> **rope_theta = 1e6 的意义**：base 越大，高维（低频）分量的周期越长。base=1e4 时最低频维度的周期约 2π×10⁴ ≈ 6.3 万；base=1e6 时约 628 万。**周期越长，长距离上的位置区分度衰减越慢**，所以想支持长上下文的模型普遍把 base 调大。

### YaRN：训练时短、推理时长

直接把模型用在超过训练长度的输入上，RoPE 会遇到没见过的角度，效果崩塌。YaRN 的做法是**按频率分段处理**：

- **高频维度**（转得快、管局部相对位置）—— 不动，因为局部关系在长文本里没变。
- **低频维度**（转得慢、管全局位置）—— 除以缩放因子 `factor=16`，等于把位置「压缩」回训练时见过的范围。
- **中间维度** —— 用 `ramp` 线性过渡，避免突变。

代码里就是 `freqs = freqs * (1 - ramp + ramp / factor)` 这一行，`beta_fast=32 / beta_slow=1` 划定过渡区的两端。**MiniMind 把它做成推理期开关**（`--inference_rope_scaling`），无需重训。

## 2.4 GQA、QK-Norm、KV Cache

| 方案 | Q 头 | KV 头 | KV Cache | 取舍 |
| --- | ---: | ---: | ---: | --- |
| **MHA** 多头 | 8 | 8 | 1.00× | 表达力最强，缓存最大 |
| **GQA** 分组查询 ← MiniMind | 8 | 4 | **0.50×** | 几乎无损，缓存减半 |
| **MQA** 多查询 | 8 | 1 | 0.125× | 缓存最小，质量损失明显 |

> **为什么减的是 KV 而不是 Q**：因为**推理时被缓存下来的只有 K 和 V**。Q 每步都是新算的、用完就丢，缓存里根本没有它。所以想压缩显存就只能砍 KV 头。**GQA 是「几组 Q 头共用一份 KV」**：MiniMind 里 8 个 Q 头分 4 组，每组 2 个 Q 头共享一份 KV，代码就是 `repeat_kv(xk, n_rep=2)`。

```python
# model/model_minimind.py · class Attention.forward · L109–135
xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
xq = xq.view(bsz, seq_len, 8, 96)     # Q: 8 头
xk = xk.view(bsz, seq_len, 4, 96)     # K: 4 头
xq, xk = self.q_norm(xq), self.k_norm(xk)   # QK-Norm，在 RoPE 之前
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

if past_key_value is not None:              # KV Cache 拼接
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)

xk = repeat_kv(xk, self.n_rep)              # 4 头 → 8 头

if self.flash and (seq_len > 1) and (past_key_value is None) and (mask is None or all(mask==1)):
    output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
else:
    scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(96)   # ← O(S²) 显存
```

> **QK-Norm：容易被忽略的现代设计**
>
> 在 Q、K 上各挂一个 `RMSNorm(head_dim)`，**放在 RoPE 之前**。作用是把 Q·K 内积的尺度钉住，避免训练中后期注意力 logits 爆大导致 softmax 饱和、进而 loss spike。这是 ViT-22B 与 Chameleon 之后被广泛采用的稳定性技巧。**被问「你怎么防 loss spike」时，这是一个源码里就有的现成答案。**

> **⚠ 真实踩坑**
>
> 上面那个 `if self.flash and ...` 的条件很苛刻：**只要带了 KV Cache（`past_key_value is not None`），或者 attention_mask 里有 0，就会掉进 `else` 分支的朴素实现**，显存随序列长度**平方**增长。
>
> 本项目在 Agentic RL 阶段实测：多轮对话累积到 `S=2500`、组大小 4 时，**单层的 scores 矩阵就要 763 MB**（`B×H×S²×4 B`），8 层加反向直接 OOM。降到 `S=1280`、组大小 2 后是 100 MB 才跑得动。**「为什么开了 Flash Attention 还 OOM」—— 答案就在这个 if 条件里。**

### KV Cache：为什么它是推理提速的关键

```
不带 Cache：生成第 n 个 token 要重算前 n−1 个的 K/V → 总计算量 O(n²)
带 Cache  ：第 n 步只算新 token 的 K/V，旧的直接取 → 总计算量 O(n)

Cache 显存 = 2 (K和V) × layers × kv_heads × head_dim × seq × dtype
           = 2 × 8 × 4 × 96 × seq × 2 B  =  12 KB / token
```

MiniMind 的 KV Cache 用最朴素的 `torch.cat` 实现（`xk = cat([past_k, xk])`），每步都重新分配显存。生产级实现会预分配一整块 buffer 按位写入（PagedAttention 更进一步做分页），但对 64M 模型这点开销可以忽略。**能说出「朴素 cat 会反复分配、生产环境要预分配」，就说明你真读过这段代码。**

## 2.5 SwiGLU 前馈网络

```python
# model/model_minimind.py · class FeedForward · L137–147
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

```
普通 FFN:  down( SiLU( up(x) ) )              2 个矩阵
SwiGLU  :  down( SiLU( gate(x) ) ⊙ up(x) )    3 个矩阵，⊙ 是逐元素相乘

其中 SiLU(x) = x · sigmoid(x)
```

**直觉类比**：`up(x)` 算出「候选内容」，`gate(x)` 算出「每一维该放行多少」，两者相乘等于给内容装了一道**逐维阀门**。相比固定的激活函数，阀门开度是随输入变化的，表达力更强。

代价是参数多了 50%（3 个矩阵而非 2 个），所以主流做法是把中间维压到约 `8/3 × hidden` 来抵消。MiniMind 用 `ceil(768×π/64)×64 = 2432`，约 **3.17 倍**，比 8/3≈2.67 略宽。

## 2.6 MoE 与负载均衡

```python
# model/model_minimind.py · class MOEFeedForward.forward · L148–175
scores = F.softmax(self.gate(x_flat), dim=-1)              # [N, 4]
topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1)    # top-1 路由
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)

# 负载均衡辅助损失
load = F.one_hot(topk_idx, num_experts).float().mean(0)    # 各专家实际命中率
self.aux_loss = (load * scores.mean(0)).sum() * num_experts * 5e-4
```

> **aux_loss 在防什么**
>
> 路由器如果发现「把所有 token 都送给专家 2」能让 loss 下降得最快，它就会这么干 —— 结果其余 3 个专家永远拿不到梯度，**MoE 退化成一个 dense 模型，白白多占 3 倍显存**。
>
> `aux_loss = Σ(实际命中率 × 平均路由概率)`，当分布完全均匀时取最小值。它**同时惩罚「命中多」和「打分高」**，因此路由器无法靠只提高分数而不实际路由来钻空子。

> **怎么证明专家没坍缩（硬证据）**
>
> 只看 aux_loss 稳定是**间接**证据 —— 它是个标量，稳定只说明损失没恶化。直接做法是在 `self.gate` 上挂前向钩子取出路由 logits，复原 top-k 分配并逐层统计。
>
> 本项目实测（留出集 320 条 × 340 token）：4 个专家占比 **25.9% / 25.0% / 24.9% / 24.3%**，归一化熵 **1.000**，8 层全部均匀，零个未使用专家。**面试时给出这组数字，比说「aux_loss 很稳」强一个量级。**（脚本见 `evals/expert_routing.py`）

## 2.7 权重绑定 tie_word_embeddings

```python
# model/model_minimind.py · MiniMindForCausalLM.__init__ · L237–241
_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
...
if self.config.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight
```

> **省了多少，为什么合理**
>
> 省下 **4,915,200 个参数**，占 63.91M 的 **7.7%** —— 对小模型是可观的比例。
>
> 合理性在于两者语义对偶：嵌入矩阵的第 *i* 行是「token *i* 的向量表示」，输出投影的第 *i* 列是「判断当前状态有多像 token *i*」。**本来就该是同一组向量。**
>
> **副作用**：绑定后梯度从两条路汇到同一张表上，等效学习率变高，有时需要略调 lr。另外统计参数量时**只能算一次**，算两次就会得到错误的 68.8M。

---

# 第三章 · 训练与优化机制

## 3.1 自回归 Loss 与因果掩码

```python
# model/model_minimind.py · MiniMindForCausalLM.forward · L249–253
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

```
输入:  [BOS]  今天   天气   很好   [EOS]
logits: p₀     p₁     p₂     p₃     p₄        ← 掐掉最后一个 p₄
labels: [BOS]  今天   天气   很好   [EOS]      ← 掐掉第一个 [BOS]

配对:  p₀→"今天"  p₁→"天气"  p₂→"很好"  p₃→"[EOS]"
```

**为什么要错一位**：第 t 个位置的输出预测的是第 t+1 个 token。不做这个 shift，模型就会被训练成「预测自己」—— 而它本来就能看到自己，loss 会瞬间趋零，模型什么也学不到。**这是新手写训练循环最经典的 bug。**

> **因果掩码在哪**：掩码**不在 loss 里，在注意力里**。`F.scaled_dot_product_attention(..., is_causal=True)` 或朴素路径的 `scores += triu(-inf, 1)` 保证位置 t 只能看到 ≤ t。
>
> 两者分工：**因果掩码防「偷看未来」，shift 保证「预测的是下一个」**。缺任何一个模型都学不成语言模型。

## 3.2 Pretrain 与 SFT 的唯一本质差别

两个阶段用**完全相同的模型、完全相同的交叉熵**。差别只有一处：**labels 里哪些位置被设成 `-100`。**

| | PretrainDataset | SFTDataset |
| --- | --- | --- |
| 输入构造 | `[BOS] + text + [EOS] + pad` | `apply_chat_template(对话)` |
| labels | `input_ids.clone()` | 全 `-100`，再挖出 assistant 段 |
| `-100` 的位置 | 仅 padding | **padding + system + user + 所有格式 token** |
| 计 loss 的比例 | ≈ 100% | ≈ 30–50% |
| 学到什么 | 语言分布本身 | 「在这种上下文里该怎么回答」 |

```python
# dataset/lm_dataset.py · SFTDataset.generate_labels · L91–105
self.bos_id = tokenizer(f'{bos_token}assistant\n').input_ids   # 回答起点标记
self.eos_id = tokenizer(f'{eos_token}\n').input_ids

labels = [-100] * len(input_ids)             # ① 先全部屏蔽
while i < len(input_ids):
    if input_ids[i:i+len(bos_id)] == bos_id:  # ② 扫描到 assistant 开头
        start = i + len(bos_id)
        while end < len(input_ids) and input_ids[end:end+len(eos_id)] != eos_id:
            end += 1
        for j in range(start, min(end+len(eos_id), max_len)):
            labels[j] = input_ids[j]          # ③ 只把回答段填回去
```

> **高频面试题的标准答案：「SFT 时如何只对 Answer 计算 loss？」**
>
> 把 labels 初始化为全 `-100`，用 **token id 序列匹配**（不是字符串匹配）定位每一段 `<|im_start|>assistant\n` 到 `<|im_end|>` 之间的区间，只把这些区间的 label 填回真实 token id。`F.cross_entropy(ignore_index=-100)` 会自动跳过其余位置。
>
> **三个加分细节**：
> 1. 多轮对话有多个 assistant 段，要循环扫描**全部**而不是只找第一个；
> 2. `<|im_end|>` 本身要**计入** loss，否则模型学不会停下来；
> 3. 匹配必须在 token id 层面做，因为同样的文字在不同上下文可能切成不同的 token。

> **⚠ 本仓库的一个真实陷阱**
>
> `pre_processing_chat` 与 `post_processing_chat` 内部调用了 `random`：以一定概率随机添加 system prompt、以 80% 概率随机移除空的 think 标签。
>
> 后果是**同一条样本每次取出来都可能不同**。本项目做多模型对比时，一开始给每个模型各迭代一次 DataLoader，结果各模型吃到的是**不同版本的数据**，配对检验的前提被破坏。修法是把 batch 固化成一份张量再喂给所有模型。**做任何「同数据对比」之前，务必确认 Dataset 是确定性的。**

## 3.3 学习率调度

```python
# trainer/trainer_utils.py · get_lr · L40–41
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

```
step = 0        → lr × (0.1 + 0.45×2) = lr × 1.00
step = 总步数/2  → lr × (0.1 + 0.45×1) = lr × 0.55
step = 总步数    → lr × (0.1 + 0.45×0) = lr × 0.10
```

这是一条**从 1.0 余弦衰减到 0.1 的曲线，不是衰减到 0**，而且**没有 warmup**。留 10% 的底是为了让模型在训练末期仍有微调能力；没有 warmup 则是因为模型只有 8 层、又有 QK-Norm 和 Pre-Norm 兜底，初期不易发散。

**被问到时要说清这两点与教科书写法的差异** —— 照搬「cosine to zero + linear warmup」的回答说明没读代码。

## 3.4 混合精度 · 梯度累积 · 梯度裁剪

```python
# trainer/train_full_sft.py · train_epoch · L14–38
with autocast_ctx:                          # bf16 自动混合精度
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps   # ① 先除，梯度才是平均值

scaler.scale(loss).backward()               # ② 放大 loss 防 fp16 下溢

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)              # ③ 裁剪前必须先还原
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

> **三个顺序不能错的点**
>
> 1. **`loss / accumulation_steps` 必须在 backward 之前** —— 梯度是累加的，不除就等于把学习率放大了 N 倍。
> 2. **`unscale_` 必须在 `clip_grad_norm_` 之前** —— 否则裁剪的是被放大过的梯度，阈值完全失去意义。这是混合精度 + 裁剪组合最经典的错误。
> 3. **`zero_grad(set_to_none=True)`** 比置零省一次显存写，且能让未使用参数的梯度真正为 `None`。

> **等效批量**：`有效 batch = batch_size × accumulation_steps × GPU 数`。本项目 MoE 预训练用 `16 × 16 = 256`，SFT 用 `6 × 3 = 18`。**梯度累积用时间换显存，数学上等价于大 batch，但 BatchNorm 类算子除外**（Transformer 用的是 LayerNorm/RMSNorm，逐样本归一化，所以完全等价）。

## 3.5 DPO 直接偏好优化

RLHF 的经典三段式是：训奖励模型 → 用 PPO 采样优化 → 反复调。DPO 的洞察是：**如果奖励模型采用 Bradley-Terry 形式，那么最优策略与奖励之间存在闭式关系，可以把奖励模型完全消掉**，直接在偏好数据上做监督式优化。

```
L_DPO = −log σ( β · [ (log π_θ(y_w|x) − log π_ref(y_w|x))
                     − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )

  y_w = chosen（更好的回答）   y_l = rejected（更差的回答）
  π_θ = 正在训练的策略        π_ref = 冻结的 SFT 模型
```

**直觉类比**：括号里是「策略相对参考模型，在好回答上提升了多少」减去「在坏回答上提升了多少」。我们希望前者大于后者 —— 也就是**把概率质量从坏回答搬到好回答上**。σ 与 −log 把它变成一个可导的分类损失。

**减去 π_ref 是关键**：只看 π_θ 的话，模型可以把两个回答的概率**一起**压低来降低 loss（灾难性遗忘）；减去参考项后只有**相对**变化才算数。

```python
# trainer/train_dpo.py · dpo_loss · L34–50
ref_log_probs = (ref_log_probs * mask).sum(dim=1)        # 只累加 assistant 段
policy_log_probs = (policy_log_probs * mask).sum(dim=1)

chosen_policy = policy_log_probs[:batch_size//2]         # 前一半是 chosen
reject_policy = policy_log_probs[batch_size//2:]         # 后一半是 rejected

pi_logratios  = chosen_policy - reject_policy
ref_logratios = chosen_ref - reject_ref
loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
```

| | PPO (RLHF) | DPO |
| --- | --- | --- |
| 需要奖励模型 | 要，且要单独训 | 不要 |
| 需要 Critic | 要（价值网络） | 不要 |
| 需要在线采样 | 要（rollout） | 不要，离线数据即可 |
| 显存里的模型数 | 4（actor/critic/ref/reward） | 2（policy/ref） |
| 稳定性 | 超参敏感 | 接近监督学习 |
| 能力上限 | 可超越数据分布 | 受限于成对数据覆盖 |

> **β 的作用（高频追问）**
>
> β 控制**策略允许偏离参考模型多远**，等价于 KL 约束的强度倒数。
>
> - **β 小（如 0.01）**：约束松，策略可以大幅偏离 → 学得快，但容易过拟合偏好数据、丢失通用能力甚至胡言乱语。
> - **β 大（如 0.5）**：约束紧，策略贴着参考模型 → 稳，但几乎学不动。
> - 常用 **0.1**，也是 MiniMind 的默认值。
>
> 从 loss 形式看，β 是 logsigmoid 输入的缩放因子：β 越大，同样的 logratio 差距产生的梯度越饱和，实际更新越小。

> **⚠ 本项目的实测结论**
>
> 在 64M 这个规模上 **DPO 完全没有效果**：权重相对变化仅 0.0055%（低于 fp16 存储精度 0.098%，等于什么都没改），200 题基准复读率 46.3% vs 基线 46.0%（p=0.66），奖励模型打分 −1.59 vs 基线 −1.52（置信区间重叠）。**两个独立指标一致指向「什么也没发生」。**
>
> 可能原因：lr 过低、β 过大、或成对数据的偏好信号在该规模下不足以产生有效梯度。**面试时能说出「我验证过它无效，并且用两个正交指标交叉确认」，比说「我用了 DPO」有价值得多。**

## 3.6 LoRA 低秩适配

```
W' = W + ΔW = W + B·A        A ∈ ℝ^(r×d),  B ∈ ℝ^(d×r),  r ≪ d

参数量:  d×d  →  2×d×r      d=768, r=16 时: 589,824 → 24,576 (4.2%)
初始化:  A ~ N(0, 0.02²),  B = 0  →  训练起点 ΔW = 0，模型行为不变
```

**直觉类比**：全量微调是「把整张 768×768 的表全改一遍」；LoRA 是「不动原表，另外记一张**薄薄的修正表**」，而这张修正表被强制写成两个瘦矩阵的乘积，秩最多只有 16。假设是：适配某个下游任务所需的权重改动，本身就是低秩的。

```python
# model/model_lora.py · apply_lora · L22–34
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and module.in_features == module.out_features:
        lora = LoRA(module.in_features, module.out_features, rank=rank)
        setattr(module, "lora", lora)
        original_forward = module.forward
        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)     # 猴子补丁，不改模型定义
        module.forward = forward_with_lora
```

> **⚠ 这个仓库特有的坑 —— 极佳的面试谈资**
>
> 注意那个筛选条件：`in_features == out_features`，**只给方阵挂 LoRA**。在 MiniMind 的结构里逐个对照：
>
> | 模块 | 形状 | 挂得上？ |
> | --- | --- | --- |
> | `q_proj` | 768 → 768 | ✅ |
> | `o_proj` | 768 → 768 | ✅ |
> | `k_proj` / `v_proj` | 768 → **384** | ❌ GQA 导致 |
> | `gate_proj` / `up_proj` | 768 → **2432** | ❌ |
> | `down_proj` | 2432 → 768 | ❌ |
>
> **所以 LoRA 实际只作用在 `q_proj` 和 `o_proj` 上**，每层 2 个、共 16 个模块。
>
> 算一下：`16 × (768×16 + 16×768) = 393,216 ≈ 0.39M`，占 63.91M 的 **0.62%** —— 与训练日志完全吻合。
>
> **这是无心还是有意？** 值得注意的是，LoRA 原论文的主要消融结论正是「只改 W_q 和 W_v 效果就很好」，而 MoE/FFN 层通常不是适配的关键。但这里漏掉了 `v_proj`（因为 GQA 让它不是方阵），**严格说与论文推荐并不一致**。能指出这一点，说明你读代码到了实处。

> **QLoRA 是什么**：QLoRA = **4-bit 量化的基座 + fp16 的 LoRA 适配器**。基座冻结所以可以激进量化（NF4 数据类型 + 双重量化），只有 LoRA 分支保持高精度参与梯度。再配合 paged optimizer 把优化器状态换出到内存，能在单张 24GB 卡上微调 65B 模型。**MiniMind 本身没实现 QLoRA**（64M 模型没必要），但这是必答题。

## 3.7 GRPO / CISPO / PPO 与 Agentic RL

| 算法 | 优势估计 | 需要 Critic | 本项目实测 |
| --- | --- | --- | --- |
| **PPO** | GAE（时序差分） | 要 | 两次独立训练均退化，见下 |
| **GRPO** | **组内相对**：同 prompt 采 G 个回答，用组内均值方差标准化 | 不要 | 复读率 45.3% → 26.4% |
| **CISPO** | 同 GRPO，裁剪方式不同 | 不要 | 复读率 45.3% → 27.2% |

> **GRPO 为什么能省掉 Critic**
>
> PPO 需要 Critic 来估计「这个状态的期望回报」，作为基线来降低策略梯度的方差。GRPO 换了个更省事的基线：**对同一个 prompt 采样 G 个回答，直接用这一组的均值当基线**、标准差做归一化。
>
> 好处是省掉一整个价值网络（显存与训练成本都降）；代价是**组内样本数 G 太小时基线噪声很大**。本项目在 Agentic RL 阶段被迫用 G=2（显存所限），日志里的 `GrpStd` 抖动明显，这是必须在报告里标注的方法学代价。

> **⚠ PPO 的两种退化（真实观测）**
>
> 本项目的 PPO 与上游官方发布的 PPO 权重，**两次完全独立的训练，产生了两种截然不同的退化**：
>
> - **本项目**：91% 的采样输出在 `</think>` 之后**为空** —— 模型把内容留在思考段，答案留空。因为在该奖励模型眼里，空答案（−0.98）竟然比 64M 模型真写出来的答案（−1.17）得分更高。
> - **官方权重**：200 道题只产出 **44 种不同开头**，其中两种占了 105 道 —— 无论问什么都回同一篇「平衡技术与伦理」的散文。
>
> **而这两个模型在复读率指标上分列全场第 1 和第 2 名。** 这是「指标很好但模型是坏的」最生动的实例。

---

# 第四章 · 数据处理管线

## 4.1 Tokenizer 与 Byte-level BPE

Tokenizer 的职责是把字符串双向映射成整数序列。**Byte-level BPE** 的构建过程：

1. **从字节开始**：初始词表是 256 个字节值。这一步保证**永远不会出现 UNK** —— 任何字符（含 emoji、生僻字）最差也能拆成字节。
2. **统计相邻对频率**：在语料上数哪两个相邻单元一起出现得最多。
3. **合并最高频对**，作为新词加入词表。
4. **重复**到词表达到目标大小（MiniMind 是 6400）。

> **6400 词表的取舍**
>
> **好处**：embedding 层只要 6400×768 = 4.92M（大模型词表 15 万，同样宽度要 115M）；输出 logits 是 `[B, S, 6400]` 而非 `[B, S, 150000]`，训练显存与算力都省一个量级。
>
> **代价**：同一段中文被切成**更多** token（词表小则合并少），等价于有效上下文变短、每字推理步数更多。
>
> **README 里的一句关键提醒**：PPL 是按 token 统计的，**跨 tokenizer 比较 PPL 没有意义**，这种情况下 BPB（Bits Per Byte）才有可比性。这是个很容易被问倒的细节。

## 4.2 ChatML 模板

```
<|im_start|>system
你是一个知识丰富的AI助手。<|im_end|>
<|im_start|>user
水的沸点是多少？<|im_end|>
<|im_start|>assistant
100摄氏度。<|im_end|>
       ↑ SFT 的 loss 只覆盖这一段（含结尾的 im_end）
```

特殊 token：`bos = <|im_start|>`，`eos = <|im_end|>`，`pad = <|endoftext|>`。模板还支持 `tools`（工具调用，渲染成 `<tools>` XML 块）与 `open_thinking`（思考段开关）。

> **⚠ 本项目踩过的坑**
>
> RL 阶段用 `RLAIFDataset` 构造 prompt 时会传 `open_thinking`（由 `--thinking_ratio` 控制概率）。**后来复现奖励曲线时我自己拼 chat 模板，漏了这个参数**，模型因此不输出 `</think>`，规则奖励里与 think 相关的两项（合计最高 +1.25）全部拿不到，重建值整体偏低约 1.4，方向都错了。
>
> **教训**：任何需要复现训练时行为的评测，**必须复用训练时的 Dataset 类**，不要自己拼模板。而抓住这个错误的是预先设的**校准对照**（拿一个日志完整的模型去验证重建流程）。

## 4.3 四种 Dataset 对比

| 类 | 返回 | 关键处理 | 用于 |
| --- | --- | --- | --- |
| `PretrainDataset` | `(input_ids, labels)` | 拼 BOS/EOS，pad 位置置 `-100` | `train_pretrain` |
| `SFTDataset` | `(input_ids, labels)` | `generate_labels` 只保留 assistant 段 | `train_full_sft` / `train_distillation` |
| `DPODataset` | 6 个张量 | chosen/rejected 各自的 x/y/mask，**已在此处做好 shift** | `train_dpo` |
| `RLAIFDataset` | `{'prompt'}` | 只给 prompt（`conversations[:-1]`），回答留给模型采样 | `train_grpo` / `train_ppo` / `train_opd` |

> **一个容易混淆处**：`SFTDataset` 返回的 `labels` **没有**提前 shift，shift 在模型 `forward` 里做；而 `DPODataset` 返回的 `x/y` **已经**错开一位了（`input_ids[:-1]` 与 `input_ids[1:]`）。两条路线的约定不同，**混用会静默地错一位**，loss 看着正常但模型学歪。

---

# 第五章 · 实测参照数据

> 以下全部是本项目在**单张 RTX 5060 8GB** 上的真实测量，不是上游 README 转述。合计 86.4 GPU 小时，六个训练阶段零崩溃零重拉。

## 5.1 资源与耗时参照表

| 阶段 | 步数 | batch × 累积 | seq | s/步 | 耗时 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE 预训练 | 79,390 × 2 | 16 × 16 | 340 | 0.185 | 8.3 h | 7.2 GB |
| MoE SFT | 150,953 × 2 | 6 × 3 | 768 | 0.174 | 14.6 h | 7.0 GB |
| 离线蒸馏 | 56,608 | 16 × 1 | 340 | 0.276 | 4.3 h | 6.2 GB |
| OPD 在线蒸馏 | 19,502 | 1 × 1 | 512+256 | 2.459 | 13.3 h | 4.5 GB |
| PPO | 19,502 | 1 × 1 | 768+256 | 1.353 | 7.3 h | 7.1 GB |
| Agentic RL | 39,988 | 1 × 1 | ≤1300 | 3.564 | 38.5 h | 7.8 GB |
| GRPO / CISPO | 19,502 | 1 × 1 | 768+256 | 3.33 | 18.0 h | 6.9 GB |

> **读这张表的方法**：**监督训练（预训练/SFT/蒸馏）单步在 0.2–0.3 秒，RL 单步 1.4–3.6 秒 —— 差一个数量级。** 原因是 RL 每步要先**自回归生成**几百个 token（几百次前向），而监督训练只要一次前向一次反向。
>
> 所以做 RL 的时间预算应该按「生成的总 token 数」估，而不是按步数估。

## 5.2 收敛参照：loss 降到多少算好

| 阶段 | 起点 | 收敛 | 怎么判断 |
| --- | ---: | ---: | --- |
| 预训练（dense） | ≈8.76 | ≈1.87 | 初值 ≈ ln(6400)=8.76，即均匀猜测 |
| 预训练（MoE） | ≈8.76 | ≈1.96 | 末 100 步均值，σ=0.165 |
| SFT（MoE） | 1.98 | 1.58 | 末 30 步均值 |

> **一个必须知道的锚点**
>
> 训练刚开始时 loss 应该约等于 **ln(vocab_size) = ln(6400) = 8.76** —— 这是模型对全词表均匀猜测的交叉熵。
>
> **如果开局 loss 远大于 8.76**，说明初始化或 label 构造有问题；**如果远小于**，八成是标签泄漏（比如忘了 shift）。**这是排查训练脚本最快的第一个检查点。**

> **⚠ 不要拿单个 batch 的 loss 下结论**
>
> 本项目实测：MoE 预训练最后 100 个采样点里，单点 loss 在 **1.58 到 2.41** 之间摆动（σ=0.165）。我一度根据末尾单点的 1.7075 得出「MoE 击败了 dense 的 1.87」，**随后被 30 点均值 1.9440 推翻**。
>
> **正确做法**：取窗口均值并报出标准差。更进一步，训练 loss 本身不能跨阶段比 —— 要比就在同一批留出数据上重新算。

## 5.3 效果对比与消融

200 题基准（10 类各 20 题，贪心解码），指标包含复读率与三道正交守卫：

| 模型 | 3-gram 复读率 | 事实准确率 | 答案为空 | 回复多样性 | 长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pretrain` | 62.7% | **0.0%** | 0.0% | **43.5%** | 530 |
| `full_sft`（基线） | 46.0% | 37.1% | 0.0% | 95.5% | 332 |
| `full_sft_moe`（教师） | 39.1% | 34.3% | 1.0% | 96.5% | 328 |
| `dpo` | 46.3% | 40.0% | 0.0% | 96.5% | 336 |
| `cispo` | 27.6% | 28.6% | 0.5% | 97.0% | 264 |
| `grpo` | 28.4% | 31.4% | 0.5% | 97.5% | 269 |
| `ppo_actor` | **7.4%** | **0.0%** | **84.0%** | 88.0% | **60** |

> **⚠ 这张表最重要的一行**
>
> **PPO 的复读率 7.4% 是全场最低（看起来最好），事实准确率却是 0.0%** —— 因为它 84% 的回答在 `</think>` 之后是空的。
>
> **只报复读率的排行榜，冠军会是一个根本没有答案的模型。** 抓住它的不是被优化的那个指标，而是「答案为空率」和「回答长度」这两个旁证。
>
> 更进一步：官方 PPO 权重**绕过了**这两道守卫（长度 417 全场最长、空答案 0%），只有**相关性**与**准确率**抓住了它。归纳出的规律是：**只看输出的指标原则上总能被某种退化绕过；把输出锚定到输入的指标才难被绕过。**

> **SFT 前后的本质差异**：看 `pretrain` 那一行 —— 准确率 0.0%、回复多样性 43.5%、长度 530。它**不是答得差，是根本不在回答**：给它一个问题，它接着往下写文章。**SFT 带来的不是「知识」，而是「对话这件事本身」**：知道该在哪停、该以什么格式回应、什么时候轮到自己说话。

## 5.4 8GB 显存工程

> **⚠ 最大的坑：显存溢出不报 OOM**
>
> 这台机器的 NVIDIA 驱动开着 **system memory fallback**：显存装不下时**不抛异常**，而是静默回落到系统内存，速度掉 4–20 倍。
>
> 实测：MoE 预训练 `bs=32` 是 **4.882 s/步**，`bs=16` 是 **0.242 s/步** —— 前者跑完要 215 小时，后者 8.5 小时。**两个都「正常运行」，代价差 25 倍。**

| 撞坑形态 | 触发方式 | 识别信号 |
| --- | --- | --- |
| batch 过大 | 预训练 bs=32 | 4.882 vs 0.242 s/步 |
| 组大小过大 | Agentic RL G=4 | 每样本 6.82 vs 1.72 s（慢 296%，非线性） |
| 序列过长 | 默认 `max_total_len=2500` | 单层 scores 763 MB → 真 OOM |
| 外部程序挤占 | 浏览器/IDE 占显存 | GPU 100% 但功耗仅 41 W |

> **可复用的判据**：**「GPU 利用率 100% + 功耗异常低 + 空闲显存不足 200 MB」** 三者同时出现，基本可判定为显存颠簸。真正在算的时候，卡的功耗应该接近 TDP。**光看利用率会被骗** —— 等 PCIe 传输也算「忙」。
>
> 解法不是估算而是实测：建模型、真跑 5 步前反向、读 `torch.cuda.max_memory_reserved()`，在「空闲显存 − 余量」下选最大可行 batch。

> **O(S²) 是可以先算后验的**
>
> 朴素注意力的单层 scores 矩阵 = `B × H × S² × 4 B`。代入不同配置：
>
> | 配置 | 单层 scores | 结果 |
> | --- | ---: | --- |
> | `S=2500, B=4` | 763 MB | OOM |
> | `S=1280, B=4` | 200 MB | 能跑但慢 |
> | `S=1280, B=2` | 100 MB | 采用 |
>
> **这三个数是先用公式算出来、再实测确认的，不是撞出来的。** 面试讲显存优化时，「我先算后验」比「我调小了 batch」高一个层次。

---

# 第六章 · 面试高频题与回答模板

> 每题给出**一句话主线**（加粗，先说结论）+ 展开。建议按主线背，展开部分理解后用自己的话讲。

### Q1 · 请用 2 分钟介绍一下你的 MiniMind 项目。
*项目介绍类*

> **我在一张 8GB 的消费级显卡上，从随机初始化开始完整走通了一个 64M 参数大模型的全生命周期：分词器、预训练、SFT、知识蒸馏、四条对齐路线，一共 86 GPU 小时、六个阶段零崩溃。**

**结构层面**是 Decoder-only：8 层、hidden 768、GQA 8 查询头配 4 个 KV 头、RoPE 位置编码 base 取 1e6、RMSNorm、SwiGLU，另外做了 MoE 版本（4 专家 top-1，198M 总参但激活只有 63.94M，与 dense 的 63.91M 几乎相同，所以两者可以公平对比）。

**但这个项目我最想讲的不是跑通了多少阶段，而是评测。** 我发现只用复读率这一个指标时，排行榜冠亚军是两个根本没有答案的坏模型 —— PPO 有 84% 的回答是空的，却拿了最低的复读率。为此我建了 200 题基准、加了准确率、相关性、答案为空率等六道正交守卫，还用多种子重训量化了训练方差，据此**主动撤回了自己之前写下的四条结论**。

**收尾**：这个项目让我具体地知道了，一个漂亮的指标在什么情况下是假的，以及该用什么去交叉验证它。

---

### Q2 · 为什么用 GQA 而不是 MHA？减的为什么是 KV 而不是 Q？
*架构细节类*

> **因为推理时被缓存的只有 K 和 V，Q 每步新算完就丢，所以要压显存就只能压 KV。**

MiniMind 用 8 个 Q 头配 4 个 KV 头，`n_rep=2`，KV Cache 直接减半。代码里就是 `repeat_kv(xk, 2)` 把 4 个 KV 头复制成 8 份去对齐 Q 头。

再往下是 MQA（所有 Q 头共用 1 份 KV），缓存能降到 1/8，但质量损失明显。**GQA 是这条线上的折中**，实践中几乎无损。

**量化一下**：MiniMind 的 KV Cache 是 `2 × 8层 × 4头 × 96 × 2 B = 12 KB/token`。如果用 MHA 就是 24 KB/token。长上下文推理时这个差距会直接决定能开多大的并发。

---

### Q3 · RoPE 是怎么把绝对位置变成相对位置的？为什么能外推？
*架构细节类*

> **RoPE 不给向量「加」位置信息，而是按位置把 Q、K 旋转一个角度；两个向量做内积时，结果只依赖它们的位置之差。**

数学上：`⟨R(m)q, R(n)k⟩ = f(q, k, m−n)`。**直觉是两根时钟指针 —— 注意力算的是夹角，而夹角只跟「差几格」有关。**

**关于外推**，要说清楚 RoPE 本身**并不天然外推**：超出训练长度后会遇到没见过的角度，效果会崩。真正让它能外推的是两件事：

1. **把 base 调大**。MiniMind 用 1e6 而非常见的 1e4，低频维度周期从约 6.3 万拉长到 628 万，长距离的位置区分度衰减更慢。
2. **YaRN 等插值方法**。按频率分段：高频维度管局部关系，不动；低频维度管全局位置，除以 factor 压回训练见过的范围；中间用 ramp 线性过渡。MiniMind 把它做成**推理期开关**，不用重训。

---

### Q4 · RMSNorm 相比 LayerNorm 省了什么？为什么内部要转 fp32？
*架构细节类*

> **省掉了「减均值」和「加偏置」两步，只保留缩放；转 fp32 是因为平方和在低精度下会溢出或丢精度。**

`LayerNorm: γ(x−μ)/√(σ²+ε)+β` ｜ `RMSNorm: γx/√(mean(x²)+ε)`。实践发现减均值那步在 Transformer 里收益很小，去掉后少一次归约、少一组参数，更快。

**fp32 那个细节值得主动说**：代码是 `(self.weight * self.norm(x.float())).type_as(x)`。混合精度下 x 是 bf16，而 768 个数的平方和在 bf16 下极易出问题，所以先升精度算完再降回去。**「归一化层内部保持 fp32」是所有主流实现的共识。**

---

### Q5 · SFT 时如何只对 Answer 计算 loss，Mask 掉 Prompt？
*训练机制类 · 最高频*

> **把 labels 初始化为全 −100，再用 token id 序列匹配定位每一段 assistant 回答，只把这些区间填回真实 id；`cross_entropy(ignore_index=-100)` 会自动跳过其余位置。**

MiniMind 的实现在 `SFTDataset.generate_labels`：以 `<|im_start|>assistant\n` 的 token 序列为起点标记，扫到 `<|im_end|>` 为终点，中间全部填回。

**三个能拉开差距的细节**：

1. 多轮对话有**多个** assistant 段，必须循环扫完，只处理第一段是常见 bug；
2. `<|im_end|>` 本身**要计入** loss，否则模型不知道该在哪停下来，推理时会一直说；
3. 匹配必须在 **token id 层面**做而非字符串层面 —— 同样的文字在不同上下文可能被切成不同的 token。

---

### Q6 · Pretrain 和 SFT 到底差在哪？为什么 SFT 之后还要 RLHF？
*训练机制类*

> **Pretrain 和 SFT 用的是同一个模型、同一个损失函数，唯一差别是 labels 里哪些位置被设成 −100；而 RLHF 引入的是交叉熵根本表达不了的相对信号和负向信号。**

**Pretrain** 每个 token 都算 loss，学的是语言分布本身。我实测过：纯预训练模型在 200 题上事实准确率 0.0%、回复多样性只有 43.5% —— 它不是答得差，是**根本不在回答**，你问它问题它接着往下写文章。

**SFT** 只对 assistant 段算 loss，学的是「对话这件事」：什么时候轮到自己说、该以什么格式回应、在哪停。

**RLHF/DPO** 的必要性在于：SFT 是模仿，它只能说「这个是对的」，永远说不出「这个比那个好」，更说不出「不要这样答」。**相对信号和负向信号是交叉熵表达不了的。**

---

### Q7 · DPO 相比 PPO 少了什么？Loss 怎么写？
*算法对比类*

> **DPO 少了奖励模型、Critic 和在线采样三样东西 —— 显存里从 4 个模型降到 2 个，训练过程接近监督学习。**

`L = −log σ( β · [ (logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l)) ] )`

**核心洞察**：如果奖励模型采用 Bradley-Terry 形式，最优策略与奖励之间有闭式关系，可以把奖励模型解析地消掉，于是 RLHF 变成了偏好数据上的一个分类问题。

**为什么必须减 π_ref**：只看 π_θ 的话，模型可以把 chosen 和 rejected 的概率**一起**压低来降 loss，那是灾难性遗忘；减去参考项后，只有**相对**变化才算数。

**代价**：DPO 受限于成对数据的覆盖范围，无法像 PPO 那样通过采样探索超出数据分布的策略。

---

### Q8 · DPO 里的 β 是干什么的？调大调小分别会怎样？
*算法对比类*

> **β 控制策略允许偏离参考模型多远，等价于 KL 约束强度的倒数。**

- **β 小（0.01）**：约束松，学得快，但容易过拟合偏好数据、丢通用能力，严重时开始胡言乱语。
- **β 大（0.5）**：约束紧，贴着参考模型，稳但学不动。
- **常用 0.1**，MiniMind 默认值也是 0.1。

**从 loss 形式看**：β 是 logsigmoid 输入的缩放因子。β 越大，同样的 logratio 差距越快进入 sigmoid 的饱和区，梯度越小、实际更新越保守。

**可以补一句实测**：我在 64M 规模上跑 DPO 发现它完全无效 —— 权重相对变化只有 0.0055%，低于 fp16 的存储精度 0.098%，等于什么都没改。两个独立指标（复读率、奖励模型打分）一致确认无变化。**这类负结果比多报一个正结果更能说明你会做验证。**

---

### Q9 · LoRA 的原理是什么？rank 怎么选？B 为什么初始化为 0？
*算法对比类*

> **LoRA 假设「适配下游任务所需的权重改动是低秩的」，于是冻结原权重 W，只训练两个瘦矩阵的乘积 ΔW = B·A。**

`W' = W + BA`，其中 `A ∈ ℝ^(r×d)`、`B ∈ ℝ^(d×r)`。d=768、r=16 时参数量从 589,824 降到 24,576，只有 **4.2%**。

**B 初始化为 0 是关键设计**：这样训练起点 ΔW = B·A = 0，模型行为与原模型完全一致，不会因为随机的适配器扰动而在训练初期崩坏。A 用高斯初始化保证有梯度流入。**如果两个都随机初始化，起点就带了一个随机扰动；两个都置零则梯度恒为零，永远学不动。**

**rank 怎么选**：任务与预训练分布越远、需要注入的新能力越多，rank 就要越大。风格适配 r=4~8 够，领域知识注入常用 16~64。

**这个仓库有个值得一提的细节**：`apply_lora` 的筛选条件是 `in_features == out_features`，只给方阵挂。在 MiniMind 里只有 `q_proj` 和 `o_proj` 是 768×768，**k_proj/v_proj 因为 GQA 变成了 768×384、FFN 是 768×2432，全都挂不上**。所以实际只有 16 个模块、0.39M 参数、占 0.62%。**这与 LoRA 原论文推荐的「W_q + W_v」并不完全一致** —— 能指出这点说明真读了代码。

---

### Q10 · 训练出现 Loss Spike 或 NaN，怎么排查？
*工程坑点类 · 高频*

> **按「先定位是数据、还是数值、还是优化」的顺序查，从最便宜的检查做起。**

**第一步：看开局 loss 对不对。** 随机初始化时 loss 应该约等于 `ln(vocab_size)`，MiniMind 是 ln(6400)=8.76。远大于说明初始化或 label 有问题；**远小于说明标签泄漏** —— 最常见的是忘了 shift，模型在预测自己。

**第二步：定位到具体 batch。** 固定随机种子复现，把爆炸前几步的数据 dump 出来。常见元凶是超长样本、全是重复字符的脏数据、或者某条样本的 label 全是 −100（导致该 batch 的 loss 是 0/0）。

**第三步：数值层面。** fp16 动态范围窄，注意力 logits 容易溢出 → 优先换 **bf16**；检查归一化层是否在 fp32 下计算；确认 `scaler.unscale_` 在 `clip_grad_norm_` **之前**调用（顺序反了裁剪就没有意义）。

**第四步：优化层面。** 降 lr、加 warmup、收紧 grad_clip。

**结构层面的预防**：MiniMind 在 Q、K 上各挂了一个 RMSNorm（**QK-Norm**），专门把注意力 logits 的尺度钉住，避免 softmax 饱和引发的 spike。加上 Pre-Norm 的干净残差通路，这类问题在这个规模上基本不出现。

---

### Q11 · 显存 OOM 怎么优化？按什么顺序试？
*工程坑点类 · 高频*

> **按「收益/代价」排序：先调不损失效果的（累积、精度、序列长度），再调有代价的（重计算、卸载）。**

1. **梯度累积**：batch 减半、累积翻倍，等效批量不变、数学等价，几乎零代价。
2. **bf16 混合精度**：激活显存直接减半。
3. **缩短序列长度**：注意力显存随 S² 增长，这一项收益最大。
4. **梯度检查点（重计算）**：用约 30% 的额外计算换掉大部分激活显存。
5. **优化器状态卸载 / 8-bit optimizer**：AdamW 每个参数要存两个 fp32 状态，是权重的两倍。
6. **LoRA / QLoRA**：直接把可训练参数砍到 1% 以下。

**但我想强调一个更前置的问题**：在我这台机器上，**显存溢出根本不报 OOM** —— 驱动开着 system memory fallback，装不下时静默回落到内存，速度掉 4 到 20 倍。实测 bs=32 是 4.882 s/步、bs=16 是 0.242 s/步，**两个都「正常运行」，跑完的时间差 25 倍。**

**识别判据**：GPU 利用率 100% + 功耗异常低（我实测 41W）+ 空闲显存不足 200MB，三者同时出现就是显存颠簸。**光看利用率会被骗，因为等 PCIe 传输也算「忙」。** 所以我写了个显存探针：真建模型、真跑 5 步前反向、读 `max_memory_reserved()`，按实测选 batch，而不是靠估算。

---

### Q12 · 开了 Flash Attention 为什么还会 OOM？
*工程坑点类*

> **因为 SDPA 的快速路径有前提条件，条件不满足时会静默回落到朴素实现，而朴素实现的显存是 O(S²)。**

看 MiniMind 的代码：`if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or all(mask==1))`。**只要带了 KV Cache，或者 attention_mask 里有 0（有 padding），就会走 else 分支**，实体化一个 `[B, H, S, S]` 的 scores 矩阵。

**我实测过这个坑**：多轮工具调用累积到 S=2500、组大小 4 时，单层的 scores 就是 `4×8×2500²×4B = 763 MB`，8 层加反向直接 OOM。把 S 降到 1280、组大小降到 2 之后是 100 MB 才跑得动。

**加分点**：这三个配置的显存我是**先用 `B×H×S²×4` 算出来、再实测确认的**，不是一次次撞出来的。而且发现组大小从 4 降到 2 时，**每样本耗时快了 296% 而不是 100%** —— 因为 O(S²) 的注意力矩阵和显存回落是复合效应，在长序列任务上 `num_generations` 根本不是线性成本参数。

---

### Q13 · MoE 的 aux_loss 是干什么的？怎么证明专家没坍缩？
*架构细节类*

> **aux_loss 防的是「路由器把所有 token 都送给同一个专家」—— 那样 MoE 会退化成 dense 模型，白占几倍显存。**

实现是 `aux_loss = Σ(实际命中率 × 平均路由概率) × num_experts × 5e-4`，分布均匀时取最小值。**它同时惩罚「命中多」和「打分高」**，所以路由器没法靠只提高分数而不实际路由来钻空子。

**但只看 aux_loss 稳定是间接证据** —— 它是个标量，稳定只说明损失没恶化，不直接说明每个专家实际分到了多少 token。**直接做法**是在 `gate` 线性层上挂前向钩子，取出路由 logits、按模型自身口径复原 top-k 分配，逐层统计。

我实测的结果是：4 个专家占比 **25.9% / 25.0% / 24.9% / 24.3%**，归一化熵 1.000，8 层全部均匀，零个未使用专家。**给出这组数字，比说「aux_loss 很稳」强一个量级。**

---

### Q14 · GRPO 为什么能省掉 Critic？代价是什么？
*算法对比类*

> **PPO 用 Critic 估计状态价值来当基线降方差；GRPO 换成「同一个 prompt 采 G 个回答，用这一组的均值当基线」。**

具体是：对每个 prompt 采样 G 个回答，各自打分后做组内标准化 `(r − mean) / std` 当作优势。**省掉一整个价值网络**，显存和训练成本都降一大截。

**代价是组内样本数 G 太小时基线噪声很大。** 我在 Agentic RL 阶段因为显存所限被迫用 G=2，日志里的 `GrpStd` 抖动明显，这是必须在结论里标注的方法学代价 —— 那条结果不应该和 G=6 的 GRPO 并排当同等口径比较。

**顺带说 PPO 的实际表现**：我和上游官方权重两次完全独立的 PPO 训练，产生了两种截然不同的退化 —— 我的是 91% 采样输出答案为空，官方的是 200 道题只有 44 种不同开头、无论问什么都回同一篇散文。**结论是 PPO 在这个代码库和规模下不稳定**，而两个坏模型在复读率上恰好排全场第 1 和第 2 名。

---

### Q15 · 你怎么判断一个评测指标是不是被「刷」了？
*方法学类 · 区分度最高*

> **看这个指标有没有「锚定到输入」。只审视输出长什么样的指标，原则上总能被某种退化绕过。**

我在这个项目里抓到过两个「指标很好但模型是坏的」的例子，而且它们的坏法完全不同：

- **例一**：我的 PPO 复读率 7.4% 全场最低，但 84% 的回答在 `</think>` 之后是空的，事实准确率 0.0%。**抓住它的是「回答长度」和「答案为空率」**。
- **例二**：官方 PPO 权重复读率 12.2% 全场第二，长度 417 全场最长、空答案率 0% —— **把上面两道守卫全绕过了**。但它 200 道题只产出 44 种不同开头，无论问什么都回同一篇 AI 伦理散文。

**归纳出的规律**：复读率、长度、空答案率、多样性，全都只看输出的形状；相关性（答案有没有提到问题里被比较的两个对象）和准确率则把输出**与输入对照**。后两个才是难被绕过的。

**再补一条更重要的**：统计显著性保护不了你。我那个「复读率降低 78.9%、p<0.0001」是完全真实的测量、完全错误的解读。**p 值只保证你没被随机性骗到，不保证你量对了东西。**

---

### Q16 · 你怎么确定实验结论不是运气？
*方法学类*

> **要分别量化两种不确定性 —— 评测噪声和训练噪声，而大多数人只量了前一半。**

**评测噪声**：换一批题目考，分数会怎么波动。用**配对自助法**（同一批题上比较，重采样 10000 次）。配对能消掉题目难度带来的方差，比独立比较敏感一个量级。

**训练噪声**：同配置只换随机种子重训一遍，模型会差多少。**这一半几乎没人量，因为很多训练框架根本没给做重复实验的接口** —— MiniMind 的种子就是硬编码 42 的，我加了 `--seed` 参数才做得了。

**实测结果里有个我完全没预料到的发现**：训练方差是**指标的属性**，不是模型的属性。同一批权重，复读率的种子间标准差只有 **0.55pp**，事实准确率却有 **4.36pp** —— 相差 8 倍。所以不能从一个指标外推到另一个。

**拿它去重新定级**：我的核心结论（策略优化降复读 18pp）在保守上界下仍有 9.3 倍标准差，稳；而两条小效应结论只有 1–2 倍标准差，**我据此撤回了它们的显著性表述**。

---

### Q17 · 自建项目的结论怎么保证不是自娱自乐？
*方法学类*

> **找外部参考实现做对照。在我加这一步之前，项目里所有数字都是自指的 —— 我的模型对比我的另一个模型。**

这留下一个从没排除过的可能：**如果我的 SFT 基线本身就训坏了，那所有「蒸馏无改善」「对齐不提准确率」的零结果，可能只是烂基线的产物。**

上游发布了官方权重，其中有一对是用**和我完全相同的 mini 数据**训的。我把它们拉下来跑同一套 200 题基准，结果同数据条件下我的 SFT 复读率低 6.6pp、准确率高 22.8pp —— **管线是好的。**

**但这一步还带出一个我没料到的发现**：官方用全量数据（8 倍量）训的模型，比 mini 数据版复读率低 29.8pp、准确率高 28.6pp。**换数据一项的收益，超过我全部方法收益之和。** 这给报告加了一条此前完全看不见的限制：我所有的方法结论都只在数据受限的体制下成立。

---

### Q18 · 为什么 MoE 的总参 198M 但你说它和 64M 的 dense 可比？
*架构细节类*

> **因为 top-1 路由下每个 token 只走 1 个专家，实际激活参数是 63.94M，与 dense 的 63.91M 几乎相同 —— 单步前向的计算量是可比的。**

算一下：MoE 每层把 dense 的 MLP（5.6M）换成 gate（3072）+ 4 个专家（4×5.6M）。总参因此涨到 198.42M，但 `num_experts_per_tok=1` 意味着每个 token 只激活其中一个专家。

`激活 = 8层 × (注意力 1.77M + 单专家 5.6M + gate) + embed 4.92M = 63.94M`，日志打印的 `198.42M-A63.94M` 就是这个意思。

**不公平的地方要主动说**：全部专家都要常驻显存，所以 MoE 权重文件是 407MB 而 dense 只有 131MB；推理延迟我实测是 3.28 s/题 vs dense 的 2.0 s/题（专家路由的 scatter/gather 有开销）。

**所以准确的表述是**：在同等激活计算量下，MoE 用 3 倍显存换来了留出集困惑度 12% 的下降。

---

### Q19 · 预训练 loss 降到多少算收敛？怎么判断？
*训练机制类*

> **先看起点对不对：随机初始化时 loss 应该约等于 ln(vocab_size)，MiniMind 是 ln(6400)=8.76。收敛值则要看词表大小和数据，不能跨项目比。**

我这里的参照：预训练降到 **1.87–1.96**，SFT 降到 **1.58**。

**但比数值更重要的是怎么读它**。我实测 MoE 预训练最后 100 个采样点，**单点 loss 在 1.58 到 2.41 之间摆动，σ=0.165**。我一度根据末尾单点的 1.7075 得出「MoE 赢了 dense 的 1.87」，随后被 30 点均值 1.9440 推翻。

**所以：① 报窗口均值和标准差，不报单点；② 训练 loss 不能跨阶段或跨模型比** —— 要比就在同一批留出数据上重新算。我为此专门写了个脚本，让所有模型跑完全相同的固化 batch。

**还有一个坑**：跨 tokenizer 时连 PPL 都不能比，因为它是按 token 统计的，词表不同 token 数就不同。那种情况要用 BPB（Bits Per Byte）。

---

### Q20 · 这个项目的局限是什么？你会怎么改进？
*收尾类 · 考察诚实度*

> **最大的局限是没有人工或强模型评判 —— 我所有指标都是程序化的，复读率只是生成质量的粗糙代理。**

**其余几条**：

1. **方差估计只有 n=2/n=3**，点估计可信但上界很宽（0.3–1.9pp）。
2. **所有方法结论只在数据受限体制下成立** —— 官方全量数据模型比 mini 数据好 29.8pp，超过我全部方法收益之和，换到数据充足的设定，方法之间的相对关系可能完全不同。
3. **单语言、单领域**，全是中文通用对话，没覆盖英文、代码、长上下文。
4. 模型本身能力很弱，事实准确率只有 37% —— 这是 64M 参数的固有限制。

**如果给我更多资源，优先级是**：先把评测做实（加 LLM-as-judge、扩带格式约束的题目），而不是把模型加大。**在评测工具还查不出 5pp 差异的时候加大模型，只会得到更多「不显著」。**

---

## 附：一页速查

| 要点 | 一句话 |
| --- | --- |
| 起点 loss | ≈ ln(vocab) = ln(6400) = **8.76**，偏离说明有 bug |
| shift | `logits[:-1]` 对 `labels[1:]`，忘了就是预测自己 |
| SFT mask | labels 全 −100，只填回 assistant 段（含 im_end） |
| GQA | 压 KV 不压 Q，因为只有 KV 进缓存；`n_rep = 8/4 = 2` |
| QK-Norm | Q/K 各挂 RMSNorm，在 RoPE 之前，防 logits 爆炸 |
| RoPE | 内积只依赖 m−n；base=1e6 拉长低频周期；YaRN 推理期外推 |
| RMSNorm | 不减均值无偏置；内部转 fp32 再降回 |
| SwiGLU | `down(SiLU(gate)⊙up)`，3 个矩阵，中间维 2432 ≈ 3.17× |
| tie_embeddings | 省 4.92M（7.7%）；统计参数量时只算一次 |
| MoE | 198.42M 总参 / 63.94M 激活；aux_loss 系数 5e-4 |
| LR 调度 | 余弦从 1.0 衰减到 **0.1 不到 0**，且**无 warmup** |
| 累积顺序 | 先除 accum → backward → `unscale_` → clip → step |
| DPO | β=0.1 控偏离；减 π_ref 防两边一起压低 |
| LoRA | B 初始化为 0；本仓库只挂方阵 → 仅 q_proj/o_proj，0.39M |
| Flash 回落 | 带 KV Cache 或 mask 有 0 → 走朴素路径，显存 O(S²) |
| sysmem fallback | 不报 OOM 只降速；判据 = 100% 利用率 + 低功耗 + 显存贴顶 |
| 指标可信度 | 只看输出的指标会被绕过；要有锚定输入的指标 |
| 两种噪声 | 评测噪声（配对自助法）+ 训练噪声（多种子重训） |

---

## 相关脚本

本手册第五章引用的所有实测数据，都可以用仓库里的评测套件复现：

| 脚本 | 作用 |
| --- | --- |
| `evals/eval_bench200.py` | 200 题 / 10 类生成基准 |
| `evals/analyze_bench200.py` | 配对自助法 + 符号检验 + 分类别拆解 |
| `evals/score_correctness.py` | 准确率、指令遵循、相关性、回复多样性四道守卫 |
| `evals/expert_routing.py` | MoE 专家路由的逐层命中分布与熵 |
| `evals/paired_ppl_test.py` | 留出集 PPL 的配对显著性检验 |
| `evals/reward_curve_rebuild.py` | 从检查点重建奖励曲线（含校准对照） |
| `evals/seed_variance_runner.py` · `seed_variance_rl.py` | 多种子重训，量化训练方差 |
| `probe_mem.py` | 显存探针，实测选 batch size |

详见 [`evals/README.md`](../evals/README.md)。
