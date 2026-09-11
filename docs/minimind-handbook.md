# MiniMind 源码手册

> 从一个 token 走完整条训练链路
>
> 骨架 hidden 768 · 8 层 · vocab 6400 ｜ 注意力 GQA 8Q/4KV · head_dim 96
> 参数量 dense 63.91M / MoE 198.42M-A63.94M ｜ 实测环境 单卡 RTX 5060 8GB · 86.4 GPU 小时

这份手册把 MiniMind 的每一处设计拆到源码行，再拆到它背后的数学。每个公式按**「直觉 → 公式 → 逐符号 → 源码对照」**四段展开，第一次学也能跟下来。所有参数量、张量形状、超参默认值都取自本仓库实际代码，不是从论文或博客转述的通用知识。

第八章的实测数据全部来自本仓库在单张 RTX 5060 8GB 上的真实训练，合计 86.4 GPU 小时，零崩溃零重拉。

**覆盖九个训练阶段**：Tokenizer → 预训练 → SFT → DPO → PPO / GRPO / CISPO → **离线蒸馏** → **OPD 在线蒸馏** → **Agentic RL** → LoRA。其中蒸馏两章（第六章）与 Agentic RL、rollout 引擎（§5.12–5.13）是本次补齐的部分。

---

## 目录

**第一章 · 全局架构** — [1.1 九阶段生命周期](#11-全生命周期九个阶段) ｜ [1.2 目录与模块职责](#12-目录结构与模块职责) ｜ [1.3 一个 token 的旅程](#13-一个-token-的完整旅程)

**第二章 · 模型结构** — [2.1 配置与参数量核算](#21-配置速查与参数量核算) ｜ [2.2 RMSNorm](#22-rmsnorm-vs-layernorm) ｜ [2.3 RoPE 与 YaRN](#23-rope-旋转位置编码) ｜ [2.4 GQA·QK-Norm·KV Cache](#24-gqa--qk-norm--kv-cache) ｜ [2.5 SwiGLU](#25-swiglu-前馈网络) ｜ [2.6 MoE](#26-moe-与负载均衡) ｜ [2.7 权重绑定](#27-权重绑定-tie_word_embeddings)

**第三章 · 预训练与 SFT** — [3.1 在优化什么](#31-语言模型到底在优化什么) ｜ [3.2 交叉熵与困惑度](#32-交叉熵困惑度与那个-876) ｜ [3.3 Teacher Forcing](#33-teacher-forcing-与那个错位一格) ｜ [3.4 SFT 标签构造](#34-sft唯一的改动是标签) ｜ [3.5 学习率调度](#35-学习率调度) ｜ [3.6 混合精度·累积·裁剪](#36-混合精度--梯度累积--梯度裁剪)

**第四章 · LoRA** — [4.1 低秩假设](#41-低秩假设从哪来) ｜ [4.2 前向与反向](#42-前向与反向梯度到底流去哪) ｜ [4.3 显存账](#43-显存账省的到底是什么) ｜ [4.4 两处偏离](#44-本仓库的两处偏离--极佳的面试谈资) ｜ [4.5 合并与 QLoRA](#45-合并回基模与-qlora)

**第五章 · 对齐与强化学习** — [5.1 为什么需要 RL](#51-为什么-sft-之后还需要-rl) ｜ [5.2 策略梯度](#52-策略梯度定理与-reinforce) ｜ [5.3 基线与优势](#53-基线与优势函数) ｜ [5.4 重要性采样](#54-重要性采样为什么能用旧数据更新) ｜ [5.5 PPO 裁剪](#55-ppo-的裁剪为什么是-min-而不是-clip) ｜ [5.6 GAE](#56-gae优势怎么逐-token-算出来) ｜ [5.7 KL 与 k1/k2/k3](#57-kl-惩罚与-k1--k2--k3-估计量) ｜ [5.8 GRPO](#58-grpo用组内均值当基线) ｜ [5.9 CISPO](#59-cispo与-grpo-只差一行但形式完全不同) ｜ [5.10 DPO](#510-dpo把-rl-变回监督学习) ｜ [5.11 奖励函数](#511-奖励函数本仓库到底怎么打分) ｜ **[5.12 Agentic RL](#512-agentic-rl让模型学会动手而不只是说话)** ｜ **[5.13 rollout 引擎](#513-rollout-引擎为什么采样要单独抽出来)** ｜ [5.14 七种算法总对比](#514-七种算法总对比)

**第六章 · 知识蒸馏**〔本次新增〕 — **[6.1 蒸馏在教什么](#61-蒸馏到底在教什么从标准答案到整张评分表)** ｜ **[6.2 温度 T 与 T²](#62-温度-t-在做什么为什么还要乘回-t-的平方)** ｜ **[6.3 离线蒸馏逐行](#63-离线蒸馏逐行读-train_distillationpy)** ｜ **[6.4 离线蒸馏的天花板](#64-离线蒸馏的天花板exposure-bias-一点没解决)** ｜ **[6.5 OPD 在线蒸馏](#65-opd把-rl-的采样接到蒸馏上)**

**第七章 · 数据管线** — [7.1 Tokenizer](#71-tokenizer-与-byte-level-bpe) ｜ [7.2 ChatML](#72-chatml-模板) ｜ [7.3 五种 Dataset](#73-五种-dataset-对比)

**第八章 · 实测参照** — [8.1 资源与耗时](#81-资源与耗时参照表) ｜ [8.2 收敛参照](#82-收敛参照) ｜ [8.3 效果对比](#83-效果对比与消融) ｜ [8.4 8GB 显存工程](#84-8gb-显存工程)

**第九章 · 面试题** — [29 题与回答模板](#第九章--面试高频题与回答模板) ｜ [附：一页速查](#附一页速查)

---

# 第一章 · 全局架构与流程

## 1.1 全生命周期：九个阶段

MiniMind 不是「拿一个预训练模型微调」，而是**从随机初始化开始把整条链路走完**。理解这条链最重要的一点是：**每一步换的是「学什么信号」，模型结构自始至终没变。**

| 阶段 | 脚本 | 做什么 | 输入 → 输出 |
| --- | --- | --- | --- |
| **0 · Tokenizer** | `train_tokenizer.py` | 把文字变成整数。Byte-level BPE，词表 6400。一旦定死，后面所有权重都绑在这个词表上 | → `model/tokenizer.json` |
| **1 · Pretrain** | `train_pretrain.py` | 学语言本身。**每个 token 都算 loss**。学到「中文长什么样」，但完全不会对话 | `pretrain_t2t_mini.jsonl` → `pretrain_768.pth` |
| **2 · SFT** | `train_full_sft.py` | 学对话格式。同样的交叉熵，但**只对 assistant 段算 loss** | `sft_t2t_mini.jsonl` → `full_sft_768.pth` |
| **3 · DPO** | `train_dpo.py` | 学「人更喜欢哪个」。chosen/rejected 成对数据，不需要奖励模型 | `dpo.jsonl` → `dpo_768.pth` |
| **4 · 策略优化** | `train_grpo.py` `train_ppo.py` | 学「怎么拿高分」。模型自己采样、奖励模型打分、按优势更新 | `rlaif.jsonl` + 奖励模型 → `grpo_768.pth` |
| **5 · 离线蒸馏** | `train_distillation.py` | 学「老师的整张概率表」。teacher 冻结，学生同时匹配硬标签和软分布 | `full_sft_moe` + `sft_*.jsonl` → `full_dist_768.pth` |
| **6 · 在线蒸馏 OPD** | `train_opd.py` | 学生**自己采样**，老师在学生自己写的句子上逐 token 批改。**不需要奖励模型** | `rlaif.jsonl` + teacher → `opd_768.pth` |
| **7 · Agentic RL** | `train_agent.py` | 学「什么时候该调工具」。多轮 rollout，真实执行工具，**工具返回的 token 不算梯度** | `agent.jsonl` + 工具 → `agent_768.pth` |
| **8 · 轻量化** | `train_lora.py` | 只训 0.39M 参数（占 0.62%）；保存统一 `.half()`，63.91M 的权重只有 131 MB | → `lora_*.pth` |

> **这条链的核心逻辑** — 三个阶段学的是三种**信息层级**：
> - **Pretrain** 学「什么话说得通」——绝对的语言概率
> - **SFT** 学「这个问题该怎么答」——条件概率，但只有正例
> - **RL/DPO** 学「A 比 B 好」——**相对**偏好，还带负例
>
> 交叉熵天生只能表达前两种，第三种必须换损失函数，这就是为什么 SFT 之后还要对齐。
>
> **蒸馏是第四种信息层级** —— 学「老师在这个位置的整张概率表长什么样」。它比 SFT 的 one-hot 稠密得多（6400 个数 vs 1 个数），又比 RL 的标量奖励稠密得多（每 token 一张表 vs 整条序列一个数）。详见第六章。
>
> **⚠ 这九个阶段不是一条必须走完的直线。** 3–7 都是从同一个 `full_sft` 权重**并行分叉**出去的不同对齐路线 —— 本项目正是这样跑的，才能把它们放在同一起点上公平对比。真实产品里通常只选其中一两条。

## 1.2 目录结构与模块职责

| 路径 | 职责 | 关键内容 |
| --- | --- | --- |
| `model/model_minimind.py` | 模型全部定义 | Config、RMSNorm、RoPE、Attention、FeedForward、MOEFeedForward、Block、CausalLM、自实现 `generate` |
| `model/model_lora.py` | LoRA 注入 | `apply_lora` 用猴子补丁改写 `forward`，不改模型定义 |
| `dataset/lm_dataset.py` | 四种数据集 | **标签与 mask 的差异全在这里** |
| `trainer/train_*.py` | 各阶段训练循环 | 每个文件自带 `train_epoch` 与 argparse，彼此独立 |
| `trainer/trainer_utils.py` | 公共工具 | `get_lr`、`setup_seed`、`lm_checkpoint`、`SkipBatchSampler` |
| `trainer/rollout_engine.py` | RL 采样引擎 | 把策略推理与训练解耦，可换 SGLang |

> **读码顺序**：先 `model_minimind.py`（一个文件读懂整个模型），再 `lm_dataset.py`（读懂标签怎么造），最后随便挑一个 `train_*.py`。**不要从 `train_*.py` 开始** —— 它们最容易懂也最没信息量。

## 1.3 一个 token 的完整旅程

batch=1、seq_len=512 的一次前向。**面试时能把形状说对，比背概念有说服力得多。**

| 步骤 | 算子 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| 输入 | `input_ids` | `[1, 512]` | 整数 token id，0–6399 |
| 嵌入 | `embed_tokens` | `[1, 512, 768]` | 查表；与 lm_head 共享权重 |
| ×8 层 | `input_layernorm` | `[1, 512, 768]` | RMSNorm，**Pre-Norm** |
| | `q/k/v_proj` | `768 / 384 / 384` | Q 8 头、KV 各 4 头 → GQA |
| | `q_norm / k_norm` | `[1,512,8,96]` | **QK-Norm**，在 RoPE 之前 |
| | `apply_rotary_pos_emb` | 同上 | 位置信息在此注入 |
| | `repeat_kv(n_rep=2)` | `[1,512,8,96]` | KV 头复制 2 份对齐 Q 头 |
| | SDPA / 朴素注意力 | `[1, 8, 512, 96]` | 朴素路径会实体化 `[1,8,512,512]` |
| | `o_proj` + 残差 | `[1, 512, 768]` | 再过 norm → SwiGLU → 残差 |
| 输出 | `norm → lm_head` | `[1, 512, 6400]` | 每个位置对全词表的 logits |
| **损失** | `shift + cross_entropy` | 标量 | 见 §3.3 |

> **显存直觉**：最后那步 `[1, 512, 6400]` 看着不大，但训练时 batch=16、seq=768 就是 `16×768×6400×4 B ≈ 300 MB`（fp32），反向还要留一份。**词表维度的 logits 往往是小模型训练里最大的单块激活。**

---

# 第二章 · 模型结构逐件拆解

## 2.1 配置速查与参数量核算

| 配置项 | 值 | 为什么是这个值 |
| --- | ---: | --- |
| `hidden_size` | 768 | 模型宽度 |
| `num_hidden_layers` | 8 | 浅网络训练快，768 的宽度又不至于模式崩溃 |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数 → `n_rep = 2`，标准 GQA |
| `head_dim` | 96 | `768 / 8` |
| `vocab_size` | 6400 | 极小词表，省 embedding 与 logits 显存 |
| `intermediate_size` | 2432 | `ceil(768×π/64)×64` —— 约 3.17 倍扩张比再对齐到 64 |
| `rope_theta` | 1e6 | 比常见的 1e4 大 100 倍，低频周期更长 |
| `tie_word_embeddings` | True | 输入嵌入与输出投影共享，省 4.92M |
| `num_experts / per_tok` | 4 / 1 | MoE 时生效 |

### 参数量核算（可手工验算，与日志逐位吻合）

```
每层 = 注意力 1,769,664 + MLP 5,603,328 + 两个 RMSNorm 1,536 = 7,374,528
  ├ 注意力 = q 768×768 + k 768×384 + v 768×384 + o 768×768 + qk_norm 192
  └ MLP    = gate 768×2432 + up 768×2432 + down 2432×768

dense = 8 × 7,374,528 + embed 4,915,200 + final_norm 768 = 63,912,192 ≈ 63.91M
MoE   = 8 × 24,187,584 + 4,915,200 + 768 = 198,416,640 ≈ 198.42M
激活   = top-1 只走 1 个专家 → 8 × 7,377,600 + 4,915,200 + 768 = 63,936,768 ≈ 63.94M
```

`embed_tokens` **只计一次** —— 因为 `tie_word_embeddings=True`。日志打印的 `198.42M-A63.94M` 就是这个意思：总参 198M，但每个 token 实际只激活 63.94M，**与 dense 的 63.91M 几乎相同 —— 这正是 MoE 对比实验成立的前提。**

## 2.2 RMSNorm vs LayerNorm

```
LayerNorm:  y = γ · (x − μ) / √(σ² + ε) + β    要算均值、方差，还有偏置 β
RMSNorm:    y = γ · x / √(mean(x²) + ε)        不减均值、无偏置
```

**直觉**：LayerNorm 是「先把这排数移到以 0 为中心，再缩放到标准长度」；RMSNorm 省掉了移动那一步，**只做缩放**。实践发现减均值在 Transformer 里收益很小，去掉后少一次归约、少一组参数。

```python
# model/model_minimind.py · class RMSNorm · L47–56
def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
    #                              ^^^^^^^^^     ^^^^^^^^^^
```

> **为什么内部转 fp32**：混合精度下 `x` 是 bf16，而 `x.pow(2).mean()` 是 768 个数的平方和，在 bf16 下**极易溢出或损失精度**。所以先升到 fp32 算归一化，再降回原精度。**「归一化层内部保持 fp32」是所有主流实现的共识。**

> **Pre-Norm**：`hidden + self_attn(input_layernorm(hidden))` —— 归一化在**子层之前**，残差是干净的恒等路径。好处是梯度沿残差直通；代价是表示尺度随层数累加，所以最后额外加了 `self.norm` 收口。

## 2.3 RoPE 旋转位置编码

绝对位置编码把位置向量*加*到输入上，模型学到的是「第 5 个位置」这种绝对概念，换个长度就失效。RoPE 的思路完全不同 —— **不加任何东西，而是把 Q 和 K 按位置「转一个角度」。**

```
把 head_dim 的 96 维两两配对成 48 个平面，第 i 个平面的旋转角：

    θ_i = pos / base^(2i/d)        base = rope_theta = 1e6

二维旋转：
    [q'_2i  ]   [cos θ  −sin θ] [q_2i  ]
    [q'_2i+1] = [sin θ   cos θ] [q_2i+1]

关键性质（RoPE 的全部意义）：
    ⟨R(m)·q , R(n)·k⟩ = f(q, k, m − n)
```

**直觉**：每个维度对是一个时钟指针，位置越靠后转得越多。两个 token 做注意力时算的是两根指针的**夹角** —— 而夹角只取决于「差了几个位置」。**绝对方式编码、相对方式生效。**

```python
# model/model_minimind.py · precompute_freqs_cis · L58–86
freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
```

> **为什么是 cat 而不是 interleave**：数学上配对的是 `(x₀,x₁), (x₂,x₃)…`，但代码里 `rotate_half` 配的是 `(x₀, x₄₈), (x₁, x₄₉)…` —— **前半段与后半段配对**。两种配法数学等价（只是维度的置换），但**切片比交错快得多**，所以 GPT-NeoX / LLaMA 系全用这种。**换实现时若两边配法不一致，权重就废了。**

> **rope_theta=1e6 与 YaRN**：base 越大，低频分量周期越长。base=1e4 时最低频周期约 6.3 万；1e6 时约 628 万。
>
> **YaRN 外推**：按频率分段 —— 高频（管局部）不动；低频（管全局）除以 `factor=16` 压回训练见过的范围；中间用 ramp 过渡。代码就是 `freqs * (1 - ramp + ramp/factor)`，做成**推理期开关**，无需重训。

## 2.4 GQA · QK-Norm · KV Cache

| 方案 | Q 头 | KV 头 | KV Cache | 取舍 |
| --- | ---: | ---: | ---: | --- |
| **MHA** | 8 | 8 | 1.00× | 表达力最强，缓存最大 |
| **GQA** ← MiniMind | 8 | 4 | **0.50×** | 几乎无损，缓存减半 |
| **MQA** | 8 | 1 | 0.125× | 缓存最小，质量损失明显 |

> **为什么减 KV 不减 Q**：因为**推理时被缓存的只有 K 和 V**。Q 每步新算完就丢，缓存里根本没有它。**GQA = 几组 Q 头共用一份 KV**：8 个 Q 头分 4 组，每组 2 个共享一份，代码就是 `repeat_kv(xk, 2)`。
>
> 量化：`KV Cache = 2 × 8层 × 4头 × 96 × 2 B = 12 KB/token`，MHA 则是 24 KB。长上下文时这直接决定并发数。

```python
# model/model_minimind.py · Attention.forward · L109–135
xq, xk = self.q_norm(xq), self.k_norm(xk)   # QK-Norm，在 RoPE 之前
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)   # KV Cache
xk = repeat_kv(xk, self.n_rep)                       # 4 头 → 8 头

if self.flash and (seq_len>1) and (past_key_value is None) and (mask is None or all(mask==1)):
    output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
else:
    scores = (xq @ xk.transpose(-2,-1)) / math.sqrt(96)   # ← O(S²) 显存
```

> **QK-Norm：容易被忽略的现代设计**
>
> 在 Q、K 上各挂 `RMSNorm(head_dim)`，**放在 RoPE 之前**。作用是把 Q·K 内积的尺度钉住，避免训练中后期注意力 logits 爆大导致 softmax 饱和、进而 loss spike。ViT-22B 与 Chameleon 之后被广泛采用。**被问「怎么防 loss spike」时，这是源码里现成的答案。**

> **⚠ 真实踩坑：开了 Flash 也会 OOM**
>
> 那个 `if` 条件很苛刻：**只要带 KV Cache，或 attention_mask 里有 0，就掉进 else 分支的朴素实现**，显存随序列长度**平方**增长。
>
> 本项目 Agentic RL 实测：`S=2500`、组大小 4 时**单层 scores 就要 763 MB**（`B×H×S²×4B`），8 层加反向直接 OOM。降到 `S=1280`、组大小 2 后是 100 MB 才跑得动。

## 2.5 SwiGLU 前馈网络

```
普通 FFN:  down( SiLU( up(x) ) )              2 个矩阵
SwiGLU  :  down( SiLU( gate(x) ) ⊙ up(x) )    3 个矩阵，⊙ 逐元素相乘

SiLU(x) = x · sigmoid(x)
```

**直觉**：`up(x)` 算「候选内容」，`gate(x)` 算「每一维放行多少」，相乘等于给内容装了一道**逐维阀门**。相比固定激活函数，阀门开度随输入变化，表达力更强。

代价是参数多 50%，所以主流把中间维压到约 `8/3 × hidden` 抵消。MiniMind 用 2432，约 **3.17 倍**。

## 2.6 MoE 与负载均衡

```python
# model/model_minimind.py · MOEFeedForward.forward · L148–175
scores = F.softmax(self.gate(x_flat), dim=-1)              # [N, 4]
topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1)    # top-1
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)
    if mask.any():
        y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)

load = F.one_hot(topk_idx, num_experts).float().mean(0)    # 各专家命中率
self.aux_loss = (load * scores.mean(0)).sum() * num_experts * 5e-4
```

> **aux_loss 在防什么**：路由器若发现「把所有 token 都送给专家 2」能让 loss 降得最快，它就会这么干 —— 其余 3 个专家永远拿不到梯度，**MoE 退化成 dense，白占 3 倍显存**。
>
> `aux_loss = Σ(实际命中率 × 平均路由概率)`，分布均匀时取最小。它**同时惩罚「命中多」和「打分高」**，所以路由器没法靠只提高分数而不实际路由来钻空子。

> **怎么证明没坍缩（硬证据）**：只看 aux_loss 稳定是**间接**证据。直接做法：在 `gate` 上挂前向钩子取出路由 logits，复原 top-k 分配并逐层统计。
>
> 本项目实测（320 条 × 340 token）：4 个专家占比 **25.9 / 25.0 / 24.9 / 24.3%**，归一化熵 **1.000**，8 层全均匀，零个未使用专家。脚本见 `evals/expert_routing.py`。

## 2.7 权重绑定 tie_word_embeddings

> 省下 **4,915,200 个参数**，占 63.91M 的 **7.7%**。
>
> 合理性在于两者语义对偶：嵌入矩阵第 *i* 行是「token *i* 的向量表示」，输出投影第 *i* 列是「当前状态有多像 token *i*」—— **本来就该是同一组向量**。
>
> **副作用**：梯度从两条路汇到同一张表，等效学习率变高；统计参数量时**只能算一次**，算两次会得到错误的 68.8M。

---

# 第三章 · 预训练与 SFT

## 3.1 语言模型到底在优化什么

这是全部训练的地基。一句话：**语言模型学的是「一段文字出现的概率」，而它把这个概率拆成了一连串「下一个词是什么」的乘积。**

**第 1 步 · 目标：给一整句话打一个概率**

```
P("今天天气很好") = ?
```

直接建模整句话的概率不可行 —— 可能的句子有无穷多种，没法列表。

**第 2 步 · 用链式法则拆开（恒等变形，没有任何近似）**

```
P(x₁…x_T) = P(x₁) · P(x₂|x₁) · P(x₃|x₁x₂) · … = ∏_{t=1..T} P(x_t | x_<t)
```

于是「给整句打分」变成了「反复回答：看了前面这些字，下一个字是什么」。**这就是自回归。** 模型每个位置输出 6400 维 logits，softmax 后就是 `P(x_t | x_<t)`。

**第 3 步 · 最大似然：让训练语料的概率最大**

```
max_θ ∏_t P_θ(x_t | x_<t)
```

连乘会下溢（几百个小于 1 的数相乘），所以取对数变成连加。

**第 4 步 · 取对数、加负号 → 变成最小化**

```
L(θ) = − (1/T) Σ_t log P_θ(x_t | x_<t)
```

这就是**负对数似然 NLL**。而它**恰好等于**交叉熵 —— 因为真实分布是 one-hot（真值那个词概率为 1、其余为 0），交叉熵 `−Σ q log p` 里只有真值那一项非零。

> **所以 `F.cross_entropy` 不是「随便选的损失」，它就是最大似然本身。**
>
> 一句话记住：**交叉熵 = 负对数似然 = 「模型对真值词给的概率有多低」的惩罚。**

## 3.2 交叉熵、困惑度与那个 8.76

```
loss = −log P(真值词)      单个位置
PPL  = exp(loss)           困惑度
```

**困惑度的直觉**：PPL = 「模型在多少个候选词之间犹豫」。PPL = 1 → 完全确定；PPL = 100 → 相当于在 100 个词里瞎猜；PPL = 6400 → 在整个词表里均匀瞎猜。

**为什么随机初始化时 loss ≈ 8.76**

```
随机权重 → 输出近似均匀分布 → P(任一词) ≈ 1/6400
loss = −log(1/6400) = log(6400) = 8.7639
PPL  = exp(8.7639) = 6400  ✓
```

> **这是排查训练脚本最快的第一个检查点。** 开局 loss 远大于 8.76 → 初始化或 label 有问题；**远小于 → 标签泄漏**（最常见是忘了 shift，模型在预测自己）。

> **⚠ 跨模型比 PPL 的陷阱**：PPL 是**按 token** 统计的。词表小 → 同样文本被切成更多 token → 每个 token 更好猜 → PPL 天然更低。**所以跨 tokenizer 比 PPL 完全没有意义。** 那种情况要用 **BPB（Bits Per Byte）**：把损失换算到「每字节多少比特」，与词表无关。

## 3.3 Teacher Forcing 与那个错位一格

```python
# model/model_minimind.py · MiniMindForCausalLM.forward · L249–253
if labels is not None:
    x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

```
输入:  [BOS]  今天   天气   很好   [EOS]
       ↓      ↓      ↓      ↓      ↓
logits: p₀     p₁     p₂     p₃     p₄     ← 掐掉最后一个 p₄（它要预测的词不存在）
labels: [BOS]  今天   天气   很好   [EOS]   ← 掐掉第一个 [BOS]（没有词预测它）

配对:  p₀→"今天"   p₁→"天气"   p₂→"很好"   p₃→"[EOS]"
```

**为什么要错一位**：第 t 个位置的输出预测的是第 t+1 个 token。不做 shift，模型就会被训练成「预测自己」—— 而它本来就能看到自己，loss 瞬间趋零。**这是新手写训练循环最经典的 bug。**

> **Teacher Forcing 是什么**
>
> 训练时，**不管模型第 t 步预测成什么，第 t+1 步喂进去的都是真实的第 t 个词**。这样 T 个位置可以**并行**算完（一次前向），而不用像推理那样串行 T 次。
>
> **代价是 exposure bias**：训练时模型看到的永远是完美的前文，推理时看到的是自己生成的（可能有错的）前文。一旦第一步错了，后面就在没见过的分布上走。**这也是 RL 和 on-policy 蒸馏存在的根本理由 —— 让模型在自己会走到的状态上学习。**

> **因果掩码在哪**：掩码**不在 loss 里，在注意力里**。两者分工：**因果掩码防「偷看未来」，shift 保证「预测的是下一个」**。缺任何一个都学不成语言模型。

## 3.4 SFT：唯一的改动是标签

Pretrain 与 SFT 用**完全相同的模型、完全相同的交叉熵**。差别只有一处：**labels 里哪些位置被设成 −100。**

```
Pretrain:  L = − (1/N) Σ_{所有 token} log P(x_t | x_<t)

SFT:       L = − (1/|A|) Σ_{t ∈ A} log P(x_t | x_<t)      A = assistant 段的位置集合
```

注意 **条件部分 `x_<t` 没变** —— prompt 依然完整地参与前向、依然被注意力看到，只是**不产生梯度**。模型学的是「给定这个 prompt，该回什么」，而不是「怎么把 prompt 本身写出来」。

| | PretrainDataset | SFTDataset |
| --- | --- | --- |
| 输入构造 | `[BOS] + text + [EOS] + pad` | `apply_chat_template(对话)` |
| labels | `input_ids.clone()` | 全 −100，再挖出 assistant 段 |
| −100 的位置 | 仅 padding | **padding + system + user + 所有格式 token** |
| 计 loss 比例 | ≈ 100% | ≈ 30–50% |
| 学到什么 | 语言分布本身 | 「这种上下文里该怎么回答」 |

```python
# dataset/lm_dataset.py · SFTDataset.generate_labels · L91–105
self.bos_id = tokenizer(f'{bos_token}assistant\n').input_ids   # 回答起点标记
self.eos_id = tokenizer(f'{eos_token}\n').input_ids

labels = [-100] * len(input_ids)             # ① 先全部屏蔽
while i < len(input_ids):
    if input_ids[i:i+len(bos_id)] == bos_id:  # ② 扫到 assistant 开头
        start = i + len(bos_id)
        while end < len(input_ids) and input_ids[end:end+len(eos_id)] != eos_id:
            end += 1
        for j in range(start, min(end+len(eos_id), max_len)):
            labels[j] = input_ids[j]          # ③ 只把回答段填回去
```

> **三个能拉开差距的细节**
> 1. 多轮对话有**多个** assistant 段，必须循环扫完，只处理第一段是常见 bug；
> 2. `<|im_end|>` 本身**要计入** loss，否则模型学不会停下来；
> 3. 匹配必须在 **token id 层面**而非字符串层面 —— 同样的文字在不同上下文可能切成不同的 token。

> **⚠ 本仓库的真实陷阱**：`pre_processing_chat` 与 `post_processing_chat` 内部调用了 `random`：以一定概率随机加 system prompt、以 80% 概率随机删空 think 标签。
>
> 后果是**同一条样本每次取出来都可能不同**。本项目做多模型对比时给每个模型各迭代一次 DataLoader，结果各模型吃到**不同版本的数据**，配对检验的前提被破坏。修法是把 batch 固化成张量再喂给所有模型。**做任何「同数据对比」前，务必确认 Dataset 是确定性的。**

## 3.5 学习率调度

```python
# trainer/trainer_utils.py · get_lr · L40–41
return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

```
step = 0    → lr × (0.1 + 0.45×2) = lr × 1.00
step = 一半  → lr × (0.1 + 0.45×1) = lr × 0.55
step = 结束  → lr × (0.1 + 0.45×0) = lr × 0.10
```

这是一条**从 1.0 余弦衰减到 0.1 的曲线，不是到 0**，而且**没有 warmup**。留 10% 的底让模型末期仍有微调能力；没有 warmup 是因为只有 8 层、又有 QK-Norm 与 Pre-Norm 兜底。**照搬「cosine to zero + linear warmup」的回答说明没读代码。**

## 3.6 混合精度 · 梯度累积 · 梯度裁剪

```python
# trainer/train_full_sft.py · train_epoch · L14–38
with autocast_ctx:
    loss = (res.loss + res.aux_loss) / args.accumulation_steps   # ① 先除
scaler.scale(loss).backward()                                    # ② 放大防下溢
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)                                   # ③ 裁剪前先还原
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

> **三个顺序不能错的点**
> 1. **`loss / accum` 必须在 backward 之前** —— 梯度是累加的，不除等于把学习率放大 N 倍。
> 2. **`unscale_` 必须在 `clip_grad_norm_` 之前** —— 否则裁剪的是被放大过的梯度，阈值完全失去意义。**这是混合精度 + 裁剪最经典的错误。**
> 3. **`zero_grad(set_to_none=True)`** 比置零省一次显存写。

> **等效批量**：`有效 batch = batch_size × accumulation_steps × GPU 数`。本项目 MoE 预训练 `16×16=256`，SFT `6×3=18`。**梯度累积用时间换显存，数学上等价于大 batch**（Transformer 用 LayerNorm/RMSNorm 逐样本归一化，所以完全等价；BatchNorm 才会不等价）。

---

# 第四章 · LoRA 参数高效微调

## 4.1 低秩假设从哪来

全量微调要更新全部 63.91M 参数，优化器状态还要再占两倍。LoRA 的出发点是一个观察：**把一个预训练模型适配到某个下游任务，所需的权重改动 ΔW 往往「秩很低」** —— ΔW 虽然是 768×768 的大矩阵，但它的信息量远没有 768×768 那么多。

```
全量微调:  W' = W + ΔW      ΔW 自由，秩最高 768，要存 589,824 个数
LoRA    :  W' = W + B·A     强制 rank(BA) ≤ r，只存 2×768×r 个数

           A ∈ ℝ^(r×d)   B ∈ ℝ^(d×r)   r ≪ d
           d=768, r=16 → 589,824 → 24,576（4.2%）
```

**直觉类比**：全量微调是「把整张 768×768 的表全改一遍」；LoRA 是「不动原表，另外记一张**薄薄的修正表**」，而这张修正表被强制写成两个瘦矩阵的乘积 —— 就像把一张大图压缩成「16 个基础图案 + 每个图案的权重」。

## 4.2 前向与反向，梯度到底流去哪

**第 1 步 · 前向：两条路相加**

```
h = W·x + B·(A·x)
    ↑冻结    ↑可训练
```

原始线性层照常算，LoRA 分支单独算一遍再加上去。代码里是猴子补丁 `forward = lambda x: original(x) + lora(x)`。

**第 2 步 · 反向：W 的梯度被丢弃，只有 A、B 更新**

```
∂L/∂B = (∂L/∂h) · (A·x)ᵀ
∂L/∂A = Bᵀ · (∂L/∂h) · xᵀ
∂L/∂W = 照算但不用（W.requires_grad = False）
```

> **注意反向传播依然要穿过整个网络** —— 上游的梯度必须一层层传下来才能算出 `∂L/∂h`。**所以 LoRA 省的是「优化器状态 + 梯度存储」，不是「反向传播的计算量」。** 这是个常见误解。

**第 3 步 · 初始化：B = 0 是关键设计**

```
A ~ N(0, 0.02²)     B = 0
→ 训练起点 ΔW = B·A = 0，模型行为与原模型完全一致
```

**为什么不能都随机**：那样起点就带了一个随机扰动，等于给一个训练好的模型加噪声。**为什么不能都置零**：`∂L/∂A = Bᵀ(...) = 0`，梯度恒为零，永远学不动。**所以必须一个随机、一个置零，而置零的要是输出侧的 B。**

```python
# model/model_lora.py · class LoRA + apply_lora · L5–34
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)   # 高斯
        self.B.weight.data.zero_()                       # 全零

    def forward(self, x):
        return self.B(self.A(x))     # ← 注意：没有 alpha/r 缩放

for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and module.in_features == module.out_features:
        ...
        module.forward = lambda x: original_forward(x) + lora(x)
```

## 4.3 显存账：省的到底是什么

| 项目 | 全量微调 | LoRA (r=16) | 说明 |
| --- | ---: | ---: | --- |
| 模型权重 | 63.91M × 2B | 63.91M × 2B | 一样，都要放 |
| **可训练参数** | 63.91M | **0.39M** | **0.62%** |
| **梯度** | 63.91M × 4B | **0.39M × 4B** | 只有可训练参数需要存梯度 |
| **AdamW 状态** | 63.91M × 8B | **0.39M × 8B** | 一阶+二阶动量，各 fp32 |
| 激活值 | 大 | 同样大 | **不省** —— 反向仍要穿过全网络 |

> **最大的一块是优化器状态**：AdamW 每个可训练参数要额外存**两个 fp32**（一阶动量 m、二阶动量 v），是 bf16 权重的 **4 倍**。全量微调时 `63.91M × 8B = 511 MB`，LoRA 只要 `0.39M × 8B = 3.1 MB`。
>
> **所以 LoRA 的省显存主要来自「梯度 + 优化器状态」，激活值一分不省。** 要省激活值得用梯度检查点。

## 4.4 本仓库的两处偏离 —— 极佳的面试谈资

### 偏离一：只给方阵挂 LoRA

| 模块 | 形状 | 是方阵？ | 挂得上？ |
| --- | ---: | :---: | --- |
| `q_proj` | 768 → 768 | ✓ | ✅ 挂上 |
| `o_proj` | 768 → 768 | ✓ | ✅ 挂上 |
| `k_proj` / `v_proj` | 768 → **384** | ✗ | ❌ GQA 导致 |
| `gate_proj` / `up_proj` | 768 → **2432** | ✗ | ❌ |
| `down_proj` | 2432 → 768 | ✗ | ❌ |

```
实际挂上的模块 = 每层 2 个 × 8 层 = 16 个
参数量 = 16 × (768×16 + 16×768) = 393,216 ≈ 0.39M
占比   = 0.39M / 63.91M = 0.62%   ← 与训练日志完全吻合
```

> **这是无心还是有意？** LoRA 原论文的主要消融结论正是「只改 W_q 和 W_v 效果就很好」。但这里因为 GQA 让 `v_proj` 变成了 768×384 的非方阵而被漏掉，实际改的是 **W_q 和 W_o**。**严格说与论文推荐并不一致** —— 能指出这一点，说明你读代码到了实处。

### 偏离二：没有 alpha 缩放

```
标准 LoRA:   ΔW = (α / r) · B·A     α 是超参，通常取 16 或 32
本仓库    :   ΔW = B·A              相当于 α = r，缩放恒为 1
```

> **α/r 是干什么的**：让**换 rank 时不必重调学习率**。r 变大时 B·A 的元素个数变多、乘积的典型幅度也变大，除以 r 正好抵消掉这个增长。
>
> **没有它的后果**：把 r 从 8 改到 64，等效更新幅度会跟着变化，**学习率必须重新调**。对固定 r 的单次实验没有影响，但做 rank 消融时会得到被混淆的结论。

## 4.5 合并回基模与 QLoRA

```python
# model/model_lora.py · merge_lora · L57–65
state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).half()
```

> **合并的意义**：把 `B·A` 这个 768×768 的矩阵直接加回 W，得到一个**普通的模型**。合并后**推理零额外开销** —— 不再有第二条分支要算。这是 LoRA 相比 Adapter 类方法的最大优势（Adapter 插了额外的层，永远也去不掉）。
>
> 代价是合并后就**不能再切换适配器**了。要同时服务多个任务就得保持不合并、动态加载。

> **QLoRA = 4-bit 基座 + fp16 适配器**：基座冻结所以可以激进量化 —— **NF4**（专为正态分布权重设计的 4-bit 数据类型）+ **双重量化**（连量化常数本身也量化）。只有 LoRA 分支保持高精度参与梯度。再配 **paged optimizer** 把优化器状态换出到内存。三者合起来能在单张 24GB 卡上微调 65B 模型。
>
> **MiniMind 本身没实现 QLoRA**（64M 没必要），但这是必答题。

---

# 第五章 · 偏好对齐与强化学习

> 这一章从最基本的策略梯度开始，一步步推到 PPO、GRPO、CISPO、DPO。**每个公式都对应本仓库的实际代码**，不是通用教材版本 —— 几处关键实现与教材写法有偏差，会逐一指出。

## 5.1 为什么 SFT 之后还需要 RL

```
SFT 能表达的:   「这个回答是对的」        正例，绝对
SFT 表达不了:   「A 比 B 好」              相对
             「不要这样答」              负例
             「这个答案有 0.7 分」        连续评分
```

交叉熵的形式决定了它只能**拉高某个特定序列的概率**。它没有任何位置可以放「这个序列有多好」这个标量。**而 RL 的整个框架就是围绕「用一个标量奖励去调整概率分布」建起来的。**

> **还有一个更微妙的理由**：SFT 是 **off-policy** 的 —— 它让模型模仿别人写的答案，而模型自己生成时会走到**训练里从没见过的状态**（exposure bias，见 §3.3）。
>
> RL 是 **on-policy** 的：**模型自己采样、在自己会走到的状态上被评价和修正**。这是它能解决 SFT 解决不了的问题的根本原因。

## 5.2 策略梯度定理与 REINFORCE

先建立最基本的框架。**把「生成一段回答」看成一局游戏**：模型是策略 π_θ，每一步选一个 token（动作），生成完整回答后拿到一个分数 R。目标是让期望分数最大。

**第 1 步 · 目标函数**

```
J(θ) = E_{y ~ π_θ(·|x)} [ R(x, y) ]
```

「按我的策略采样出一个回答，期望能拿多少分」。我们要**最大化**它。

**第 2 步 · 难点：期望里的分布本身依赖 θ**

```
∇_θ J = ∇_θ Σ_y π_θ(y) R(y)
```

R 是个黑盒（可能是人打分、可能是奖励模型），对 θ 不可导。而 π_θ 在求和号里面。

**第 3 步 · 对数导数技巧（log-derivative trick）**

```
∇π = π · ∇log π        因为 ∇log π = ∇π / π
```

这一步是整个策略梯度的枢纽 —— 它把「对分布求导」变回了「在分布下求期望」。

**第 4 步 · 策略梯度定理**

```
∇_θ J = E_{y ~ π_θ} [ R(y) · ∇_θ log π_θ(y) ]
```

**现在可以用采样来估计了**：采 N 个回答，算平均。

**直觉**：拿到高分的回答，就提高它的对数概率（梯度上升）；低分的就压低。**R 就是每个样本梯度的权重。**

**第 5 步 · 拆到 token 级**

```
∇_θ J ≈ (1/N) Σ_i R(y⁽ⁱ⁾) · Σ_t ∇_θ log π_θ(y⁽ⁱ⁾_t | x, y⁽ⁱ⁾_<t)
```

这就是 **REINFORCE**。写成损失就是 `loss = -(R * logp).mean()`。

> **⚠ REINFORCE 的致命问题：方差极大。**
>
> 假设所有回答的分数都在 5 到 7 之间 —— 它们全是正的，于是**所有**回答的概率都被推高，只是幅度不同。真正有用的信号（「7 分比 5 分好」）被淹没在「大家都是正分」这个共同的偏移里。
>
> 采样噪声还会让同一个回答这次采到、下次采不到，梯度方向剧烈摆动。**这直接引出下一节的基线。**

## 5.3 基线与优势函数

**第 1 步 · 减去一个不依赖动作的基线 b，期望不变**

```
E[ (R − b) · ∇log π ] = E[ R·∇log π ] − b·E[ ∇log π ]
                                          ↑ 这一项恒等于 0
```

因为 `E[∇log π] = Σ π · ∇log π = Σ ∇π = ∇(Σπ) = ∇1 = 0`。

**所以减基线不引入偏差，但能大幅降低方差** —— 这是「免费的午餐」。

**第 2 步 · 优势函数：把「绝对分」变成「相对分」**

```
A(x, y) = R(x, y) − b(x)
```

**直觉**：不问「这个回答好不好」，而问「**这个回答比平均水平好多少**」。比平均好 → A > 0 → 提高概率；比平均差 → A < 0 → 压低概率。**现在有了真正的负信号。**

**第 3 步 · 基线怎么来？—— 这是各算法分道扬镳的地方**

```
PPO  ：训一个 Critic 网络 V(s) 来预测「这个状态的期望回报」
GRPO ：同一 prompt 采 G 个回答，用这一组的均值当基线
DPO  ：干脆不采样，用成对数据直接算相对偏好
```

## 5.4 重要性采样：为什么能用旧数据更新

策略梯度要求样本来自**当前**策略 π_θ。但采样很贵（要自回归生成几百个 token），只更新一次就扔掉太浪费。**能不能用同一批样本更新好几次？**

```
E_{y ~ π_new} [ f(y) ] = E_{y ~ π_old} [ (π_new(y) / π_old(y)) · f(y) ]
                                          ↑ 重要性权重 ratio
```

**直觉类比**：你想知道「北京人的平均身高」，手上却只有上海人的数据。重要性采样说：可以用上海数据算，但每个样本要**加权** —— 在北京更常见的那类人，权重调高。

代入 RL：用旧策略采的样本，乘上 `π_new/π_old` 这个比值来修正，就能估计新策略的梯度。

```python
# trainer/train_grpo.py
ratio = torch.exp(per_token_logps - old_per_token_logps)
# exp(log a − log b) = a / b，在对数空间做除法更稳定
```

> **⚠ 重要性采样的危险**：如果 π_new 和 π_old 差太远，ratio 会变得极大或极小 —— **方差爆炸，估计完全失效**。
>
> 比如某个 token 在旧策略下概率 0.001、新策略下 0.5，ratio = 500，这一个样本就会主导整个梯度。**这就是 PPO 要「裁剪」的直接原因。**

## 5.5 PPO 的裁剪：为什么是 min 而不是 clip

```
L^CLIP = E[ min( ratio · A , clip(ratio, 1−ε, 1+ε) · A ) ]

  ratio = π_θ(a|s) / π_θ_old(a|s)      ε = clip_epsilon = 0.2
```

很多人以为「裁剪」就是把 ratio 限制在 [0.8, 1.2]。**不是** —— 如果只做 clip，当 ratio 已经超出范围时梯度就恒为 0，模型再也回不来了。**外面套一层 min 才是关键。**

| 情况 | A 的符号 | ratio | min 选中哪个 | 效果 |
| --- | :---: | :---: | --- | --- |
| 好动作，已提升很多 | A > 0 | > 1+ε | clip 项（较小） | **梯度截断**，不再继续推高 |
| 好动作，提升不多 | A > 0 | ≈ 1 | ratio 项 | 正常更新 |
| 坏动作，已压低很多 | A < 0 | < 1−ε | clip 项（更负→较小） | **梯度截断** |
| **坏动作，反而被推高了** | A < 0 | > 1+ε | ratio 项（更负） | **不截断！**让它被拉回来 |

> **最后一行是 min 的全部意义**：当策略「跑错方向跑太远」时（坏动作的概率反而涨了），**我们希望梯度继续起作用把它拉回来**，而不是因为超出裁剪区间就放弃。`min` 恰好保证了这一点。
>
> **一句话：裁剪只在「已经朝对的方向走够了」时刹车，绝不在「走错方向」时刹车。**

```python
# trainer/train_ppo.py · L209–211（用 max 等价实现）
policy_loss = torch.max(-advantages[inds] * ratio,
                        -advantages[inds] * torch.clamp(ratio, 1.0-ε, 1.0+ε))
# 注意：max(-a, -b) == -min(a, b)，加了负号所以 min 变 max，等价
```

## 5.6 GAE：优势怎么逐 token 算出来

上面说的 A 是「整个回答」的优势。但生成是逐 token 的，**我们需要知道每一个 token 的贡献**。GAE（Generalized Advantage Estimation）就是干这个的。

**第 1 步 · TD 误差：单步的「惊喜程度」**

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)
```

**直觉**：「我原本以为这局能拿 V(s_t) 分；走了一步后，实际拿到 r_t、并且新局面值 V(s_{t+1}) 分。」**δ 就是这次比预期好了多少。**

**第 2 步 · GAE：把未来所有的惊喜按 γλ 衰减加起来**

```
A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + …
    = δ_t + γλ·A_{t+1}          ← 倒着递推，一遍算完
```

**λ 控制偏差-方差权衡**：λ=0 时只看一步（偏差大、方差小）；λ=1 时看完整轨迹（无偏、方差大）。本仓库 `lam=0.95`、`gamma=1.0`。

**第 3 步 · 本仓库的奖励是「稀疏终局奖励」**

```
token_rewards = 全 0
token_rewards[最后一个 token] += 外部奖励
```

中间每个 token 都**没有**即时奖励，整段回答的分数全部记在最后一个 token 上。**GAE 的作用就是把这个终局分数合理地分摊回前面每一个 token** —— 这正是它存在的意义。

```python
# trainer/train_ppo.py · GAE 倒推 · L139–150
lastgaelam = torch.zeros(B)
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t+1] if t < gen_len-1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam    # 递推
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)
returns = advantages + old_resp_values          # Critic 的回归目标

# 优势归一化：减均值除标准差（只在有效 token 上算）
advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

> **Critic 是什么**：本仓库的 Critic 直接**复用整个 MiniMind 骨架，只把 lm_head 换成 `Linear(768, 1)`**：
> ```python
> class CriticModel(MiniMindForCausalLM):
>     self.value_head = nn.Linear(params.hidden_size, 1)
> ```
> 它的训练目标是让 `V(s_t)` 逼近 `returns_t`，损失是**带裁剪的均方误差**（防止 value 更新过猛）：
> `value_loss = 0.5·max((V−R)², (clip(V, V_old±0.2)−R)²)`

## 5.7 KL 惩罚与 k1 / k2 / k3 估计量

光追求奖励，模型会「为了高分不择手段」—— 输出胡言乱语但恰好骗过奖励模型。**KL 惩罚是拴住它的绳子**：不许离原来的 SFT 模型太远。

```
真实 KL:  KL(π_θ ‖ π_ref) = E_{y~π_θ}[ log π_θ(y) − log π_ref(y) ]

但我们只有采样点，需要一个估计量。记 r = log π_ref − log π_θ：

  k1 = −r              无偏，但方差大，还可能为负（KL 不该为负）
  k2 = r² / 2          恒非负，方差小，但有偏
  k3 = exp(r) − r − 1  恒非负、无偏、方差小 ← 三者兼得
```

> **为什么 k3 恒非负**：`e^r ≥ 1 + r` 对所有实数成立（指数函数在 r=0 处的切线），所以 `e^r − r − 1 ≥ 0`，等号仅在 r=0（两分布相同）时取到。
>
> **这是 John Schulman 提出的估计量**，现在是 GRPO/PPO 实现的事实标准。

```python
# GRPO/CISPO：加在 loss 里
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1          # k3

# PPO：同样加在 loss 里（不是加在 reward 里），系数 kl_coef=0.02
kl_ref_penalty = (torch.exp(ref-mb) - (ref-mb) - 1.0) ...
loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref_penalty

# PPO 还用 k2 做早停判据
approx_kl = (0.5 * (log_ratio ** 2) * mask).sum() / mask.sum()   # k2
if approx_kl_val > args.early_stop_kl:   # 阈值 0.25，超了就停止本轮更新
    break
```

> **两个值得说的实现细节**
>
> **① KL 加在 loss 里还是 reward 里？** 经典 RLHF 是把 KL 惩罚**加进 reward**（`r' = r − β·KL`），这样它会经过 GAE 分摊到每个 token。**本仓库是直接加在 loss 上**，实现更简单，但 KL 的信号不会参与优势估计。两种做法都常见，能说出区别是加分项。
>
> **② 早停用 k2 而非 k3**：早停只需要一个「偏离程度」的标量指标，k2 计算更省（不用 exp）。

> **⚠ DDP 死锁的坑**：代码里特意注释了：**早停必须同步各卡的 approx_kl**。否则某张卡触发 break 退出循环、其他卡还在等它参与 all-reduce，**整个训练直接卡死**。这是多卡 RL 训练的经典陷阱。

## 5.8 GRPO：用组内均值当基线

**第 1 步 · 核心思想：同一个问题，让模型答 G 遍**

```
prompt x  →  y⁽¹⁾, y⁽²⁾, …, y⁽ᴳ⁾     （本仓库 G = num_generations = 6）
          →  r⁽¹⁾, r⁽²⁾, …, r⁽ᴳ⁾     （奖励模型各打一分）
```

这 G 个回答面对的是**同一个 prompt**，难度完全一样 —— 所以它们的平均分天然就是一个**好基线**。

**第 2 步 · 组内标准化得到优势**

```
A⁽ⁱ⁾ = ( r⁽ⁱ⁾ − mean(r⁽¹⁾…r⁽ᴳ⁾) ) / ( std(r⁽¹⁾…r⁽ᴳ⁾) + 1e-4 )
```

除以标准差是为了让不同难度的 prompt 产生的优势**尺度一致**。

**第 3 步 · 损失：PPO 的裁剪 + k3 KL 惩罚**

```
L = −[ min(ratio·A, clip(ratio, 1±ε)·A) − β·KL_k3 ]
```

整段回答共享同一个 A（序列级优势），但 ratio 和 KL 是**逐 token** 的。

```python
# trainer/train_grpo.py
grouped_rewards = rewards.view(-1, args.num_generations)   # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(G)
std_r  = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(G)
advantages = (rewards - mean_r) / (std_r + 1e-4)

clipped_ratio = torch.clamp(ratio, 1-ε, 1+ε)
per_token_loss = -(torch.min(ratio*A, clipped_ratio*A) - args.beta * per_token_kl)

# 逐序列平均（按有效 token 数），再对 batch 平均
policy_loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)).mean()
```

> **⚠ G 太小的代价（本项目实测）**：组内只有 G 个样本，**用它们估均值和标准差本身就有噪声**。G 越小噪声越大，极端情况 G=2 时标准差几乎不可信。
>
> 本项目在 Agentic RL 阶段因显存所限被迫用 **G=2**，日志里的 `GrpStd` 抖动明显。**这条结果因此不应与 G=6 的 GRPO 并排当同等口径比较** —— 这类方法学代价必须主动标注。

## 5.9 CISPO：与 GRPO 只差一行，但形式完全不同

```python
# trainer/train_grpo.py · 两个分支对比
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * A * per_token_logps - β * per_token_kl)
else:  # grpo
    per_token_loss = -(torch.min(ratio*A, clamp(ratio,1±ε)*A) - β * per_token_kl)
```

```
GRPO :  L = −min( ratio·A , clip(ratio)·A )      ratio 带梯度，是优化目标的一部分
CISPO:  L = −clamp(ratio).detach() · A · log π   ratio 被切断梯度，只当权重
```

> **这是两种不同的范式**
>
> - **GRPO 是 PPO 式**的 —— 对 ratio 求导，梯度里会出现「ratio 的变化率」。
> - **CISPO 是 REINFORCE 式**的 —— 回到最原始的 `A · ∇log π` 形式，只是给它乘上一个**被 detach 的重要性权重**作为修正系数。
>
> **为什么这样做**：detach 之后 ratio 不再参与反向，梯度形式更简单、更稳定；`clamp(max=ε_high)` **只截上界**（本仓库 `epsilon_high=5.0`），防止个别样本的权重过大主导梯度，但**不截下界** —— 低概率样本的权重小本来就不危险。

> **本项目实测：两者收益相当**。200 题基准复读率：CISPO **27.6%**、GRPO **28.4%**，相差 0.75pp。
>
> 但要注意：**这个差异小于训练噪声**（多种子重训实测 σ≈0.80pp），所以正确表述是「**未能区分**」，而不是「确认无差异」。**区分这两句话的分量，是评测方法学上很值钱的一课。**

## 5.10 DPO：把 RL 变回监督学习

前面所有算法都要「采样 → 打分 → 更新」。DPO 问了一个大胆的问题：**能不能把奖励模型从公式里彻底消掉，直接在偏好数据上做监督训练？**

**第 1 步 · RLHF 的标准目标：最大化奖励，同时不要跑太远**

```
max_π  E_{y~π}[ r(x,y) ] − β · KL( π ‖ π_ref )
```

**第 2 步 · 这个优化问题有闭式解**

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

Z(x) 是归一化常数（配分函数）。**直觉**：最优策略就是「参考模型的分布，按奖励做指数加权」—— 奖励高的地方概率被放大。

**第 3 步 · 反解出 r —— 这是最关键的一步**

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β·log Z(x)
```

**奖励可以用策略表示出来！** 换句话说，**「一个策略」和「一个奖励函数」是一一对应的**，训策略等价于训奖励。

**第 4 步 · 代入 Bradley-Terry 偏好模型**

```
P(y_w ≻ y_l | x) = σ( r(x,y_w) − r(x,y_l) )
```

BT 模型是说「A 胜过 B 的概率由两者分数之差的 sigmoid 给出」。**注意这里是<u>差</u>** —— 而 `β·log Z(x)` 只依赖 x，**在相减时被完全消掉**。那个讨厌的配分函数没了。

**第 5 步 · 最大似然 → DPO 损失**

```
L_DPO = −log σ( β · [ (log π_θ(y_w) − log π_ref(y_w))
                     − (log π_θ(y_l) − log π_ref(y_l)) ] )
```

**奖励模型、Critic、采样，全部消失了。** 剩下的就是一个二分类的交叉熵，可以像监督学习一样训。

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

> **为什么必须减 π_ref**：只看 π_θ 的话，模型可以把 chosen 和 rejected 的概率**一起压低**来降低 loss —— 那是**灾难性遗忘**，模型什么都不敢说了。减去参考项后，**只有「相对于原模型的相对变化」才算数**。

> **β 的作用（高频追问）**：β 控制策略允许偏离参考模型多远，**等价于 KL 约束强度的倒数**。
> - **β 小（0.01）**：约束松，学得快，但容易过拟合偏好数据、丢通用能力。
> - **β 大（0.5）**：约束紧，贴着参考模型，稳但学不动。
> - 常用 **0.1**，本仓库默认也是 0.1。
>
> **从公式看**：β 是 logsigmoid 输入的缩放因子。β 越大，同样的 logratio 差距越快进入 sigmoid 饱和区，梯度越小、更新越保守。

> **⚠ 本项目实测：DPO 在 64M 规模完全无效**
>
> 权重相对变化仅 **0.0055%**，低于 fp16 存储精度 0.098%（**等于什么都没改**）；200 题复读率 46.3% vs 基线 46.0%（p=0.66）；奖励模型打分 −1.59 vs −1.52（置信区间重叠）。
>
> **两个独立指标一致指向「什么也没发生」。** 面试时能说出「我验证过它无效，并用两个正交指标交叉确认」，比说「我用了 DPO」有价值得多。

## 5.11 奖励函数：本仓库到底怎么打分

前面所有算法里的那个 `r`，在本仓库是**规则项 + 奖励模型**两部分之和。这部分常被忽略，但它直接决定了模型会学成什么样。

```python
# trainer/train_grpo.py · calculate_rewards · L37–68
r = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5   # 长度合理
answer = response
if '</think>' in response:
    thinking, answer = response.split('</think>', 1)
    r += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5  # 思考段长度
    r += 0.25 if response.count('</think>') == 1 else -0.25   # 只闭合一次
    answer = answer.strip()
r -= rep_penalty(answer)                       # 3-gram 复读惩罚，上限 0.5
r += reward_model.get_score(messages, answer)  # internlm2-1.8B 打分，裁到 ±3
```

| 组成 | 范围 | 作用 |
| --- | ---: | --- |
| 长度合理 | ±0.5 | 防止过短或过长 |
| 思考段长度 | +1.0 / −0.5 | 鼓励产生思考段 |
| 单次闭合 | ±0.25 | 防止乱输出多个 `</think>` |
| 复读惩罚 | 0 ~ −0.5 | 3-gram 重复率越高扣越多 |
| **奖励模型打分** | ±3 | internlm2-1.8B，**唯一评价「答得好不好」的项** |

> **⚠ 奖励函数漏洞的真实后果**
>
> 注意 `answer = answer_content.strip()` 这一行：**如果模型在 `</think>` 之后什么都不写，answer 就是空串**。空串的 `rep_penalty` 恒为 0（没有 3-gram），而规则项那 +1.75 照拿不误。
>
> 本项目的 PPO **真的找到了这个洞**：91% 的采样输出答案为空。更关键的是 —— 在这个奖励模型眼里，**空答案（−0.98）竟然比 64M 模型真写出来的答案（−1.17）得分更高**。
>
> **这不是「奖励函数写错了」，而是「奖励模型对这个能力段区分度不够」**，两者的修法完全不同。

## 5.12 Agentic RL：让模型学会「动手」而不只是「说话」

**先用大白话说清楚它和前面的区别。**

前面 5.8–5.10 的所有算法，模型的一局游戏是这样的：给一个问题 → 模型一口气写完回答 → 打分 → 结束。**模型从头到尾只是在「说话」，它说的每个字都是自己编的。**

Agentic RL 改的是这一条：**模型可以中途停下来，说一句「我要用计算器」，程序真的去把计算器跑一遍，把真实结果贴回它的上下文里，它再接着往下写。**

这就带来一个前面从没出现过的问题：**这段上下文里，有一部分字不是模型写的。**

```
模型写的：  我需要算一下 <tool_call>{"name":"calculate_math","arguments":{"expression":"23*17"}}</tool_call>
程序塞的：  <tool>{"result": "391"}</tool>          ← 这几个 token 不是模型的动作！
模型写的：  所以答案是 391。
```

**整个 Agentic RL 与普通 RL 的结构性区别，就只有这一件事。** 损失函数、优势估计、KL 惩罚，全部原封不动复用 GRPO / CISPO。

### 为什么工具返回的 token 必须屏蔽梯度

回忆 §5.2 的策略梯度：`∇J = E[ A · ∇log π(动作) ]`。这个式子里的「动作」，必须是**策略自己选的**。

工具返回的 `391` 不是模型选的，是环境给的**观测**。如果不屏蔽：

1. **教错了东西**：优势 A 是正的时候，梯度会去提高 `log π(391)` —— 等于在教模型「预测计算器会输出什么」。我们要的是它学会**什么时候调用工具**，不是学会**背下工具的答案**。
2. **污染梯度**：工具输出是确定性的。同一个表达式永远返回同一个结果，模型很快就能把它背下来，`log π` 趋近 0，这些位置贡献的梯度极小却占了归一化分母，**把真正的动作 token 的信号稀释掉**。

```python
# trainer/train_agent.py · rollout_single
response_ids.extend(new_ids)
response_mask.extend([1] * len(new_ids))          # ① 模型生成的 → mask=1，有梯度
response_old_logps.extend(new_logps)

# ... 执行工具，把结果塞进 messages ...

obs_delta = observe_ids[current_len:]             # ② 工具结果新增的 token
response_ids.extend(obs_delta)
response_mask.extend([0] * len(obs_delta))        # ← 关键：mask=0，不产生梯度
response_old_logps.extend([0.0] * len(obs_delta))
```

> **⭐ 这是 Agentic RL 最常被问的一个点。** 一句话答案：**只对「模型自己采样出来的 token」算策略梯度，环境返回的观测 token 必须 mask 掉** —— 它们不是动作，是状态的一部分。

### 多轮循环怎么组织

```
for turn in 0..2:                                    # max_turns = 3
    context = apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True)
    生成 → new_text，记下逐 token 的 logprob
    calls = parse_tool_calls(new_text)                # 正则抠 <tool_call>...</tool_call>
    if not calls: break                               # ← 没调工具就收工
    for call in calls:
        result = execute_tool(name, args)             # 真的执行（带 1 秒超时）
        messages.append({"role": "tool", "content": json.dumps(result)[:2048]})
    观测 token 追加进 response，mask 填 0
```

两个工程细节值得记：

- **`[:2048]` 截断**：`calculate_math` 用 `eval` 算表达式，模型如果写出 `9**9**9` 会得到一个几十万位的天文数字，不截断会把 tokenizer 撑爆。
- **1 秒超时 + 沙箱 eval**：`eval(expr, {"__builtins__": {}, "math": math})` 把内置函数全部清空，只留 `math`，防止模型生成的表达式执行任意代码。

### 奖励函数分两支

这是本仓库的设计，和 §5.11 的纯文本奖励不同 —— **它按「这次到底调没调工具」走两条完全不同的打分路径**。

| | 不调工具（当普通问答） | 调了工具 |
| --- | --- | --- |
| 长度分 | `5 ≤ len ≤ 800` → +0.5，否则 −0.5 | — |
| 思考分 | `20 ≤ len(think) ≤ 300` → +1.0，否则 −0.5；`</think>` 恰好一个 ±0.25 | — |
| 模型分 | 奖励模型打分 | — |
| **工具对齐分** | — | `gap = |有效调用数 − 标注数| + 多余调用数`；`gap=0` → +0.5，否则 `−0.5 × gap` |
| **结果验证分** | — | **`+2.5 × 命中的 GT 数 / GT 总数`** |
| 未完成罚 | — | 三轮还没收工 → −0.5 |
| 复读罚 | `−rep_penalty` | `−rep_penalty` |
| 总分 | clip 到 ±3.0 | clip 到 ±3.0 |

标签不匹配的扣分独立于两支之外：`−0.5 × |<tool_call> 个数 − </tool_call> 个数|`，防止模型只写半个标签骗过正则。

**结果验证分是这个奖励里最硬的一项**（2.5 分，占满分的 83%），它的判定方式值得单独说：

```python
def validate_gt_in_text(text, gt_list):
    # 两条路都算命中：① 字符串包含  ② 数值差 < 1e-6
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text)]
    return {g for g in gt_list if (str(g).lower() in text.lower())
                               or any(abs(float(g) - n) < 1e-6 for n in nums)}
```

**为什么要留数值容差这条路**：工具算出来的是数字，模型可能写成 `391`、`391.0`、`391.00`。纯字符串匹配会把后两种判错，而它们**明明是对的**。这是一个很小但很能说明问题的设计 —— **奖励函数判错，比模型答错更致命**，因为模型会朝着错误的判据优化。

### 本项目实测

`lr=3e-7`（与 GRPO / PPO 同一档，RL 阶段统一用这个量级）、`num_generations=4`、`max_turns=3`、`beta=0.1`、`thinking_ratio=0.1`，单卡 **38.5 GPU 小时**，是整个项目最长的一个阶段。

| 指标 | 基线 `full_sft` | `agent` | 变化 |
| --- | ---: | ---: | --- |
| 复读率 | 46.0% | **35.2%** | ✅ −10.8pp |
| 固定 96 题奖励 | −1.5179 | **−0.7159** | ✅ +0.80 |
| 通用问答准确率 | 37.1% | **25.7%** | ❌ −11.4pp |

> **⚠ 必须诚实报告的一条：通用问答能力下降了。**
>
> 这不是 bug，是**对齐税（alignment tax）**：训练分布是工具调用任务，评测集是通用问答，两者不是一回事。模型把容量挪去学「什么时候该发 `<tool_call>`」，通用问答就退步了。
>
> **面试时主动说出这一点，比藏着它有价值得多。** 正确的做法是**再补一个工具调用的评测集**分别看两个能力，而不是只用通用问答一把尺子量所有模型 —— 这是本项目承认没做完的部分之一。

---

## 5.13 rollout 引擎：为什么采样要单独抽出来

从 GRPO 开始，每个 RL 脚本都有这么一行：

```python
rollout_engine = create_rollout_engine(engine_type="torch", policy_model=model, ...)
```

**大白话：训练和采样是两件性质完全不同的事，硬塞在一起会互相拖累。**

| | 训练前向 | 采样（rollout） |
| --- | --- | --- |
| 调用次数 | 1 次前向 + 1 次反向 | **几百次前向**（每个 token 一次） |
| 要梯度吗 | 要 | 不要 |
| 瓶颈在哪 | 算力 | **访存**（KV Cache 反复读写） |
| 优化手段 | 混合精度、梯度累积 | KV Cache、continuous batching、PagedAttention |

这就是为什么 §8.1 的耗时表里，**监督训练单步 0.2–0.3 秒，RL 单步 1.4–3.6 秒，差一个数量级** —— RL 的时间几乎全花在采样上。

本仓库把这件事抽象成一个接口，两个实现：

- **`TorchRolloutEngine`** — 直接调模型自带的 `generate`，零依赖，本项目全程用的是这个。
- **`SGLangRolloutEngine`** — 把策略权重写到共享目录，让一个独立的 SGLang 服务去做推理，通过 HTTP 拿回结果。工业界 RLHF 的标准做法（连续批处理 + RadixAttention，吞吐能差 5–10 倍）。

### 一个容易被忽略的点：策略其实是「略微过期」的

```python
if step % args.save_interval == 0 or step == iters:
    rollout_engine.update_policy(model)      # save_interval = 10
```

**采样引擎里的权重不是每步都同步的，而是每 10 步同步一次。** 也就是说第 11 步采样用的，其实是第 10 步的权重，而模型已经走了 1 步。

严格说这不是纯 on-policy，是 **near-on-policy**。而这恰好解释了 §5.4 的重要性采样为什么不是摆设：

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
```

如果真的是完全同步的纯 on-policy，`ratio` 会恒等于 1，PPO 的裁剪就是个空操作。**正因为采样策略落后训练策略最多 10 步，`ratio` 才真的会偏离 1，裁剪才真的在起作用。**

> **面试可以这样说**：「工业界的 RLHF 不可能纯 on-policy —— 采样用的是独立的推理服务，权重同步有延迟，而且一批采样数据通常要复用好几个 minibatch。**重要性采样就是为了让这些略微过期的数据仍然能用来做无偏更新，PPO 的裁剪则是限制它别过期太多。**」

---

## 5.14 七种算法总对比

把第五、六两章的算法放在一张表里。**前四列是「用标量奖励调概率」，后两列是「用老师的分布调概率」，最后一列是「用成对偏好调概率」。**

| 维度 | PPO | GRPO | CISPO | Agentic RL | 离线蒸馏 | OPD | DPO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 学习信号 | 标量奖励 | 标量奖励 | 标量奖励 | 标量奖励 | **teacher 分布** | **teacher 分布** | 成对偏好 |
| 信号密度 | 1 数/序列 | 1 数/序列 | 1 数/序列 | 1 数/序列 | **6400 数/token** | **6400 数/token** | 1 数/对 |
| 前文来自 | 模型自己 | 模型自己 | 模型自己 | 模型自己 + **工具返回** | **数据集** | 模型自己 | 数据集 |
| 基线来源 | Critic 网络 | 组内均值 | 组内均值 | 组内均值 | 不需要 | 不需要 | 不需要 |
| 优势估计 | GAE（逐 token） | 组内标准化 | 同 GRPO | 同 GRPO | 不需要 | 不需要 | 不需要 |
| ratio 用法 | 参与目标，带梯度 | 参与目标，带梯度 | **detach，只当权重** | 随 loss_type 切换 | 无 | 无 | 无 |
| 裁剪 | 双边 1±ε | 双边 1±ε | 只截上界 ε_high | 同上 | 无 | 无 | 无 |
| KL 约束 | k3 加在 loss | k3 加在 loss | k3 加在 loss | k3 加在 loss | 无 | **k3 就是目标本身** | 隐含在 β 里 |
| 需要奖励模型 | 要 | 要 | 要 | 要 | **不要** | **不要** | **不要** |
| 需要在线采样 | 要 | 要 | 要 | 要 | **不要** | 要 | **不要** |
| 显存里的模型数 | 4 | 3 | 3 | 3 | **2** | **2** | **2** |
| **实测复读率** | 7.4%（退化） | 28.4% | 27.6% | 35.2% | 47.8% | 49.4% | 46.3% |
| **实测准确率** | 0.0%（退化） | 31.4% | 28.6% | 25.7% | 31.4% | 20.0% | 40.0% |
| **实测耗时** | 7.3 h | 18.0 h | 18.0 h | **38.5 h** | 4.3 h | 13.3 h | 短 |

> **⚠ 读这张表的三条注意事项**
>
> **① 基线是 `full_sft`：复读率 46.0%、准确率 37.1%。** 对照它才知道哪些是真的提升。
>
> **② PPO 那个 7.4% 的复读率不是最好，是坏掉了** —— 事实准确率 0.0%、84% 的答案为空。**指标最漂亮的那一列，恰恰是唯一坏掉的模型。** 详见 §8.3。
>
> **③ 复读率和准确率经常反向动。** CISPO / GRPO 把复读率砍掉一半，准确率也掉了 6–9pp；DPO 准确率最高却完全没降复读率。**没有哪个算法在所有指标上通吃 —— 报告时必须两个都给，只报一个就是在挑对自己有利的那把尺子。**

---

# 第六章 · 知识蒸馏：离线与 On-Policy

> 这一章对应 `trainer/train_distillation.py` 与 `trainer/train_opd.py`。
> **它们是本项目里唯一「有两个模型同时在显存里，但第二个不是奖励模型」的阶段。**

## 6.1 蒸馏到底在教什么：从「标准答案」到「整张评分表」

先用一个最朴素的比方。

**SFT 是这样教的**：老师说「这题选 C」。你照着记下来。

**蒸馏是这样教的**：老师把自己的完整判断给你看 ——「A 我给 3 分，B 我给 0.5 分，**C 我给 60 分**，D 我给 0.5 分」。

两种教法，正确答案都是 C。但第二种额外告诉了你一件事：**A 虽然错了，但它比 B 和 D 错得轻**。

这就是蒸馏的全部秘密。那些「错误选项上的相对分数」有个名字叫 **暗知识（dark knowledge）**，它编码了老师对这些词之间相似关系的理解。

### 为什么这能让学生学得更快

算一笔信息账：

| | 每个 token 位置，学生收到多少信息 |
| --- | --- |
| **SFT（one-hot 标签）** | 1 个数：`−log p(正确词)` |
| **蒸馏（teacher 分布）** | **6400 个数**：整个词表上的概率分布 |

一条训练样本，SFT 只告诉模型「这里该是『很』」，蒸馏还顺带告诉它「『非常』也挺合适、『但是』完全不行」。**同样的数据量，梯度里携带的信息密度差了三个数量级。**

这也是为什么蒸馏常被说成「**用更少的数据达到同样的效果**」—— 不是数据变多了，是每条数据被榨出的信息变多了。

### 两个损失长什么样

```
SFT  ：  L = − Σ_v  q(v) · log p_student(v)          q 是 one-hot，只有真值那项为 1
蒸馏 ：  L = − Σ_v  p_teacher(v) · log p_student(v)   teacher 的软分布，6400 项全非零
```

**形式完全一样，只是把 one-hot 的 q 换成了 teacher 的软分布。** 代码里写成 KL 散度（`F.kl_div`）而不是交叉熵，差的只是一个与学生无关的常数项 `Σ p_t log p_t`（teacher 的熵），对梯度没有任何影响。

---

## 6.2 温度 T 在做什么，为什么还要乘回 T 的平方

### 温度在做什么

```
带温度的 softmax:   p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

| T 的取值 | 分布变成什么样 | 直觉 |
| --- | --- | --- |
| `T → 0` | 变成 one-hot | 只认最高分那一个，退化回硬标签 |
| `T = 1` | 原本的 softmax | 不做任何改变 |
| `T = 1.5`（本仓库） | 稍微压平 | 小概率项被抬起来一些 |
| `T → ∞` | 变成均匀分布 | 所有词一样，信息全丢 |

**为什么要压平**：训练好的 teacher 往往非常自信，正确词的概率可能是 0.999，其余 6399 个词加起来才 0.001。此时暗知识全挤在小数点后好几位，学生几乎看不见。**升温就是把这些差异放大到学生能感知的范围。**

### 那个 T² 从哪来

这是 Hinton 原论文里的一个细节，也是面试里能拉开差距的点。

对学生 logits 求导时，`z/T` 这一层会带出一个 `1/T` 的因子；teacher 那一侧的软化又贡献一个 `1/T`。**净效果是梯度整体缩小到原来的 1/T²。**

如果不补偿，`T=1.5` 时蒸馏项的梯度只有 CE 项的 1/2.25，两个损失的权重 `alpha` 就失去了意义 —— 你以为在做 50:50 的混合，实际是 69:31。

```python
# trainer/train_distillation.py · distillation_loss · L25–36
teacher_probs    = F.softmax(teacher_logits / temperature, dim=-1).detach()
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
return (temperature ** 2) * kl          # ← 把 1/T² 乘回来
```

> **一句话记住**：**T 控制「暗知识被放大多少」，T² 负责「把放大带来的梯度缩小补偿回去」**，两者一起保证换 T 时不用重调学习率和 alpha。

---

## 6.3 离线蒸馏：逐行读 `train_distillation.py`

### 总损失：CE 和 KD 的加权混合

```python
loss = alpha * ce_loss + (1 - alpha) * distill_loss       # alpha = 0.5
```

| 项 | 学的是 | 权重 |
| --- | --- | --- |
| `ce_loss` | 数据集里的**真实标签**（硬标签） | 0.5 |
| `distill_loss` | **teacher 的分布**（软标签） | 0.5 |

**为什么不能只要 KD**：teacher 也会错。只学 teacher，学生的上限被牢牢钉在 teacher 上，而且 teacher 的错误会被原样继承。保留 CE 项等于同时给学生看「标准答案」，**让它至少不会比数据集更差**。

### 只在 assistant 段算 —— 与 SFT 用同一套 mask

```python
loss_mask = (labels[..., 1:] != -100).float()             # ← 与 §3.4 完全一致
distill_loss = distillation_loss(
    student_logits.view(-1, V)[loss_mask_flat == 1],      # 只挑出回答段的位置
    teacher_logits.view(-1, V)[loss_mask_flat == 1],
    temperature=temperature)
```

**这一步的意义**：prompt 部分不是模型要学的内容，如果在那里也匹配 teacher 分布，等于在教学生「怎么把用户的问题复述出来」。mask 是从 `SFTDataset` 的 `-100` 直接推出来的，两个阶段的口径天然一致。

注意 `reduction='batchmean'` 是在**已经展平并筛选过的 token 张量**上做的 —— 所以这里的「batch」实际是「有效 token 数」，结果是**逐 token 平均**，与 CE 项的归一化口径对齐。

### 教师词表对齐

```python
teacher_logits = teacher_logits[..., :vocab_size_student]
```

允许 teacher 的词表比 student 大（比如拿一个真正的大模型当 teacher）。本项目两边都是 6400，这行是空操作，但它决定了脚本的通用性。

### 本项目的配置：一个不太常规的选择

| | teacher | student |
| --- | --- | --- |
| 权重 | `full_sft_768_moe.pth` | `full_sft_768.pth` |
| 结构 | **MoE**，4 专家 top-1 | **dense** |
| 总参数 | 198.42M | 63.91M |
| **激活参数** | **63.94M** | **63.91M** |

> **⚠ 这不是常规的「大模型教小模型」。**
>
> 两者 hidden、层数、词表**完全相同**，激活参数量只差 0.03M。真正的差别只有一个：**teacher 每层有 4 个专家可以挑，student 只有 1 个固定的 FFN。**
>
> 所以这个实验问的其实是：**「稀疏容量」能不能通过蒸馏转移到「稠密容量」里去？** 这个提法比「大教小」有意思得多，也是本项目值得讲的一个设计。

`lr=5e-6`，只有 SFT 的一半（`1e-5`）—— 因为起点已经是一个训练好的 `full_sft`，这里是**微调**而不是重训。

---

## 6.4 离线蒸馏的天花板：exposure bias 一点没解决

离线蒸馏虽然换了损失，但有一件事**和 SFT 一模一样**：

**它依然是 teacher forcing（§3.3）。** 喂给学生和 teacher 的前文，都是数据集里那段**完美的、人写好的**文字。

于是问题来了：

```
训练时： [完美前文] → 学生匹配 teacher 在这里的分布  ✓
推理时： [学生自己生成的、可能已经跑偏的前文] → teacher 从来没在这种状态下示范过  ✗
```

**学生在自己会犯的错误之后该怎么补救，teacher 一次都没教过。** 这正是 §3.3 说的 exposure bias，蒸馏没有解决它，只是把老师从「数据集」换成了「另一个模型」。

### 实测：两个指标指向相反的方向

| 模型 | 留出集 PPL | 200 题复读率 | 事实准确率 |
| --- | ---: | ---: | ---: |
| `full_sft`（学生起点 / 基线） | 12.26 | 46.0% | 37.1% |
| `full_sft_moe`（teacher） | **11.20** | 39.1% | 34.3% |
| `full_dist`（离线蒸馏后） | **11.91** | 47.8% | 31.4% |

> **⭐ 这张表是全手册最值得琢磨的一张。**
>
> **PPL 变好了**（12.26 → 11.91，向 teacher 的 11.20 靠拢），**生成质量却没变好**（复读率 46.0% → 47.8%，准确率 37.1% → 31.4%）。
>
> 一点都不矛盾，因为这两件事测的根本不是一回事：
>
> - **PPL 是 teacher forcing 下测的** —— 给完美前文，问「下一个词猜得准不准」。蒸馏优化的正是这个，所以它变好完全在预期之内。
> - **复读率和准确率是自由生成下测的** —— 让模型自己往下写几百个 token，错误会累积。**这正是蒸馏没有训练过的状态。**
>
> **结论：PPL 提升 ≠ 生成质量提升。** 任何只报 PPL 的蒸馏结论都值得追问一句「自由生成下测了吗」。这也是本项目坚持**两套指标一起报**的原因。

**而这个缺口，就是下一节 OPD 要补的。**

---

## 6.5 OPD：把 RL 的采样接到蒸馏上

### 一句话版本

**离线蒸馏**：老师批改**课本上的范文**，让学生照着改。
**OPD**：**学生自己写一篇**，老师在**学生自己写的每一句话上**批注「这里我会用别的词」。

差别只有一个 —— **前文是谁写的**。

### 公式

```
L_OPD(θ) =      E        [  (1/|y|) · Σ  D( π_θ(·|s_t) ‖ ν(·|s_t) )  ]
            x ~ p_data              t
            y ~ π_θ(·|x)
                 ↑
     关键就在这里：y 是「学生自己」采样出来的

   π_θ = 学生策略（可训练）        ν = 教师策略（冻结）
   s_t = (x, y_<t)   —— 第 t 步的状态 = 问题 + 学生已写出的前 t−1 个词
```

把它和 §5.2 的 RL 目标放在一起看，结构**完全相同**：

```
RL  ：  J(θ) =        E         [  R(x, y)  ]
               x ~ data, y ~ π_θ          ↑
                                  一个标量，打在整条序列上

OPD ：  L(θ) =        E         [  (1/|y|) Σ D( π_θ ‖ ν )  ]
               x ~ data, y ~ π_θ                t      ↑
                                       每 token 一整张概率表的散度

        ↑ 采样方式完全一样（y 都来自模型自己），信号完全不同
```

> **⭐ 一句话讲清 OPD 是什么：**
>
> **OPD = 强化学习的「在线采样」结构 + 知识蒸馏的「稠密信号」。**
>
> 它从 RL 那里拿来了**在学生自己会走到的状态上学习**（解决 exposure bias），
> 从蒸馏那里拿来了**每个 token 一整张概率表的监督**（信号稠密），
> 而且**完全不需要奖励模型** —— teacher 的分布本身就是奖励信号。

对比三者：

| | 前文来自 | 信号 | 信号密度 | 要奖励模型吗 |
| --- | --- | --- | --- | --- |
| **SFT** | 数据集 | one-hot 标签 | 1 个数 / token | 不要 |
| **离线蒸馏** | 数据集 | teacher 分布 | 6400 个数 / token | 不要 |
| **RL（GRPO 等）** | **学生自己** | 标量奖励 | **1 个数 / 整条序列** | **要** |
| **OPD** | **学生自己** | **teacher 分布** | **6400 个数 / token** | **不要** |

OPD 在这张表里占了最好的那一格：**既在正确的状态分布上学，信号又最稠密，还省掉了奖励模型。** 代价是必须有一个比自己强的 teacher。

### 三种散度模式

```python
# trainer/train_opd.py · distill_divergence · L32–60
if loss_mode == "k3":                                    # ← 默认
    s_lp = log_softmax(student_logits).gather(-1, sampled_ids)   # 只取采样到的那个词
    t_lp = log_softmax(teacher_logits).gather(-1, sampled_ids)
    r = t_lp - s_lp
    return torch.exp(r) - r - 1.0                        # 恒 ≥ 0

if loss_mode == "forward_kl":                            # 全词表前向 KL
    return (t_probs * (t_logprobs - s_logprobs)).sum(-1)

if loss_mode == "forward_kl_topk":                       # 只在 teacher top-32 上算
    top_lp, top_idx = torch.topk(t_logprobs, k=32, dim=-1)
    return (top_lp.exp() * (top_lp - s_logprobs.gather(-1, top_idx))).sum(-1)
```

| 模式 | 在哪些词上算 | 显存 | 特点 |
| --- | --- | ---: | --- |
| **`k3`**（默认） | **只在采样到的那 1 个词** | 最省 | 反向 KL 的低方差无偏估计，见 §5.7 |
| `forward_kl` | 全部 6400 个词 | 最贵 | 要实体化两份 `[B, R, 6400]` 的 log_softmax |
| `forward_kl_topk` | teacher 的 top-32 | 中等 | verl 的默认，抓住 99% 的概率质量 |

**为什么默认选 k3**：8GB 单卡上，`forward_kl` 要同时放两份 `[6, 256, 6400]` 的 fp32 张量（各 37 MB），再加上反向图 —— 而 k3 只需要 `gather` 出 `[6, 256]` 两个向量。**这是被显存逼出来的选择，但恰好也是 §5.7 那个方差最小的估计量。**

注意这里 k3 的 `r = t_lp − s_lp`，与 §5.7 RL 里的 `r = ref_logp − policy_logp` **形式完全一致** —— teacher 在 OPD 里扮演的角色，就是 RL 里 reference model 的角色，只不过在 RL 里它是「别跑太远」的约束，在 OPD 里它是**唯一的学习目标**。

### 归一化：为什么要先除以 |y| 再取平均

```python
distill_loss = ((per_token_loss * completion_mask).sum(dim=1)      # 每条序列自己求和
                / completion_mask.sum(dim=1).clamp(min=1)          # 除以自己的长度 |y|
               ).mean()                                            # 再对 batch 平均
```

**如果直接对所有 token 求和再除以总 token 数**，一条 200 token 的回答对损失的贡献会是一条 20 token 回答的 10 倍 —— 模型会被长回答主导。**先按序列归一，等于给每条采样样本同等的话语权**，这与 GRPO / CISPO 的做法（§5.8）完全一致。

`completion_mask` 还做了一件事：**截到第一个 EOS 为止**。EOS 之后的 padding 不参与任何计算。

### 工程：一条 8GB 卡上的真实约束

```python
# trainer/train_opd.py · L248–255
# 教师只产 logits 当 KL 目标，不回传梯度，半精度存放足够。
# 注意教师前向在 no_grad 里但不在 autocast 里，所以 fp32 权重会让整个
# 前向和 [B, S-1, 6400] 的 logits 都跑在 fp32 上。198M 的 MoE 教师
# 光权重就 793MB，转 bf16 后 396MB，logits 也减半。
if args.teacher_dtype != 'float32':
    teacher_model = teacher_model.to(dtype=torch.bfloat16)
```

> **这段注释值得单独讲**，因为它是一个**非常容易踩、又极难 debug** 的坑：
>
> `torch.no_grad()` 只关掉梯度，**不改变计算精度**。teacher 的前向没有被 `autocast` 包住，所以只要权重是 fp32，整条前向和那个 `[B, S−1, 6400]` 的 logits 张量就全在 fp32 上跑 —— **白白多占一倍显存，而这部分精度对最终的散度毫无影响**（下游 `t_logits.float()` 又转回来了）。
>
> 手动把 teacher 转成 bf16，权重 793MB → 396MB，logits 也减半。**在这张卡上，省下的正是仅剩的那点余量。**

另外两个与其他脚本不同的地方：

- **学习率调度用的是 `CosineAnnealingLR(eta_min=lr/10)`**，不是 §3.5 那个手写的 `get_lr`。两者形状一样（余弦衰减到 10%），但一个是 PyTorch 内置调度器、一个是每步手动改 `param_group['lr']`。
- **`batch_size=1`，`num_generations=6`** —— 一个 prompt 采 6 条，所以实际每步要生成 6×256 个 token。这就是单步 2.46 秒的来源。

### 本项目实测：一个失败的结果，和它为什么有价值

| 模型 | 留出集 PPL | 复读率 | 事实准确率 | 固定 96 题奖励 |
| --- | ---: | ---: | ---: | ---: |
| `full_sft`（学生起点） | 12.26 | 46.0% | 37.1% | −1.5179 |
| `full_sft_moe`（teacher） | 11.20 | 39.1% | 34.3% | −1.4211 |
| `full_dist`（离线蒸馏） | 11.91 | 47.8% | 31.4% | −1.6106 |
| **`opd`（在线蒸馏）** | **12.55** | **49.4%** | **20.0%** | −1.5398 |

**OPD 比离线蒸馏更差，甚至比什么都不做更差。** 13.3 GPU 小时，没有换来任何一项指标的提升。

必须诚实地报告它，并且**把原因分析清楚 —— 这比一个漂亮的结果更能证明你理解了这件事**：

**1 · teacher 本身不够强 —— 这是最根本的原因。**

teacher 的准确率只有 **34.3%**，比学生起点的 **37.1% 还低**。**蒸馏的上限是 teacher**，让一个 37.1% 的学生去对齐一个 34.3% 的老师，结果向下收敛完全符合预期 —— 而 OPD 因为信号更稠密、又直接施加在学生自己的分布上，**收敛得比离线蒸馏更彻底，所以掉得更多**。

**2 · 训练日志本身就预告了这个结果。**

```
首 100 步  distill loss 均值 = 0.2995
末 100 步  distill loss 均值 = 0.2482   （σ = 0.0665）
```

**19,502 步、13.3 GPU 小时，散度只降了 17%，而且降幅（0.05）还不到单点波动（σ=0.067）的一倍。**

这说明学生和 teacher 在 KL 意义上**本来就已经很接近了** —— 它们是同一个 `full_sft` 权重出发、只差 MoE 与否的两个模型。**没有多少「知识」可转移，自然也转移不出什么效果。** 如果当初盯着这条曲线，第一个小时就该停下来。

**3 · 选 teacher 时我只看了 PPL。**

teacher 的留出集 PPL 是全场最低的 11.20，看起来是最强的模型。但 §6.4 已经说明白了：**PPL 低不代表生成质量好**。我在选 teacher 时犯的，正是自己在 §6.4 里总结出来的那个错误。

**4 · 还有一层分布错配。** 实跑用的是默认 `thinking_ratio=0.5` —— 一半的采样样本被要求带 `<think>` 段，而 200 题基准是普通问答。训练分布和评测分布不完全一致，这会让实际效果比「同分布下」再差一点。

> **这一节的两条方法论 —— 花 13.3 GPU 小时买来的：**
>
> **① 选 teacher 要用你最终关心的那个指标，不能用代理指标。**
> 如果当初用「200 题准确率」而不是「PPL」去挑 teacher，就会发现根本**没有一个模型比 `full_sft` 更适合当老师** —— 那么正确的决策是**不做这个实验**，或者先训一个更强的 teacher 出来。
>
> **② 蒸馏开跑后，第一件要看的事是散度降不降。**
> 散度不降 = 学生和 teacher 本来就没差多少 = 没有知识可转移。**这个判断在前 100 步就能做出来，不用等 13 小时。** 它是蒸馏实验最便宜的一个早停判据。

---

# 第七章 · 数据处理管线

## 7.1 Tokenizer 与 Byte-level BPE

1. **从字节开始**：初始词表是 256 个字节值。这保证**永远不会出现 UNK** —— 任何字符（含 emoji、生僻字）最差也能拆成字节。
2. **统计相邻对频率**：在语料上数哪两个相邻单元一起出现得最多。
3. **合并最高频对**，作为新词加入词表。
4. **重复**到词表达到目标大小（MiniMind 是 6400）。

> **6400 词表的取舍**
>
> **好处**：embedding 只要 6400×768 = 4.92M（大模型词表 15 万，同宽度要 115M）；输出 logits 是 `[B,S,6400]` 而非 `[B,S,150000]`，训练显存与算力都省一个量级。
>
> **代价**：同一段中文被切成**更多** token，等价于有效上下文变短、每字推理步数更多。
>
> **关键提醒**：PPL 按 token 统计，**跨 tokenizer 比 PPL 没有意义**，那种情况要用 BPB。

## 7.2 ChatML 模板

```
<|im_start|>system
你是一个知识丰富的AI助手。<|im_end|>
<|im_start|>user
水的沸点是多少？<|im_end|>
<|im_start|>assistant
100摄氏度。<|im_end|>
       ↑ SFT 的 loss 只覆盖这一段（含结尾的 im_end）
```

特殊 token：`bos = <|im_start|>`，`eos = <|im_end|>`，`pad = <|endoftext|>`。模板还支持 `tools`（渲染成 `<tools>` XML 块）与 `open_thinking`（思考段开关）。

> **⚠ 本项目踩过的坑**：RL 阶段 `RLAIFDataset` 构造 prompt 时会传 `open_thinking`（由 `--thinking_ratio` 控制）。**复现奖励曲线时我自己拼 chat 模板，漏了这个参数**，模型因此不输出 `</think>`，规则奖励里 think 相关的两项（合计 +1.25）全部拿不到，重建值整体偏低约 1.4，**方向都错了**。
>
> **教训**：任何要复现训练时行为的评测，**必须复用训练时的 Dataset 类**。而抓住这个错误的是预先设的**校准对照**（拿一个日志完整的模型验证重建流程）。

## 7.3 五种 Dataset 对比

| 类 | 返回 | 关键处理 | 用于 |
| --- | --- | --- | --- |
| `PretrainDataset` | `(input_ids, labels)` | 拼 BOS/EOS，pad 位置置 −100 | train_pretrain |
| `SFTDataset` | `(input_ids, labels)` | `generate_labels` 只保留 assistant 段 | train_full_sft / distillation |
| `DPODataset` | 6 个张量 | chosen/rejected 各自 x/y/mask，**已在此处 shift** | train_dpo |
| `RLAIFDataset` | `{'prompt'}` | 只给 prompt，回答留给模型采样 | train_grpo / ppo / opd |
| `AgentRLDataset` | `{'messages','tools','gt'}` | **不 tokenize**，把工具定义和标准答案原样传出 | train_agent |

> **`AgentRLDataset` 为什么不 tokenize**：多轮工具调用的上下文要在 rollout 过程中**反复重建**（每执行完一次工具就要把结果拼回去再套一次 `apply_chat_template`），提前 tokenize 没有意义。它只负责把 `messages` / `tools` / `gt` 三样东西原样递出去，编码全部交给训练循环。

> **⚠ 容易混淆处**：`SFTDataset` 返回的 labels **没有**提前 shift，shift 在模型 `forward` 里做；而 `DPODataset` 返回的 x/y **已经**错开一位。**两条路线约定不同，混用会静默错一位**，loss 看着正常但模型学歪。

---

# 第八章 · 实测参照数据

> 以下全部是本项目在**单张 RTX 5060 8GB** 上的真实测量。合计 86.4 GPU 小时，六个训练阶段零崩溃零重拉。

## 8.1 资源与耗时参照表

| 阶段 | 步数 | batch × 累积 | seq | s/步 | 耗时 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MoE 预训练 | 79,390 × 2 | 16 × 16 | 340 | 0.185 | 8.3 h | 7.2 GB |
| MoE SFT | 150,953 × 2 | 6 × 3 | 768 | 0.174 | 14.6 h | 7.0 GB |
| 离线蒸馏 | 56,608 | 16 × 1 | 340 | 0.276 | 4.3 h | 6.2 GB |
| OPD 在线蒸馏 | 19,502 | 1 × 1 | 512+256 | 2.459 | 13.3 h | 4.5 GB |
| PPO | 19,502 | 1 × 1 | 768+256 | 1.353 | 7.3 h | 7.1 GB |
| Agentic RL | 39,988 | 1 × 1 | ≤1300 | 3.564 | 38.5 h | 7.8 GB |
| GRPO / CISPO | 19,502 | 1 × 1 | 768+256 | 3.33 | 18.0 h | 6.9 GB |

> **读这张表的方法**：**监督训练单步 0.2–0.3 秒，RL 单步 1.4–3.6 秒 —— 差一个数量级。** 因为 RL 每步要先**自回归生成**几百个 token（几百次前向），监督训练只要一次前向一次反向。
>
> **做 RL 的时间预算应该按「生成的总 token 数」估，而不是按步数估。**

## 8.2 收敛参照

| 阶段 | 起点 | 收敛 | 判断依据 |
| --- | ---: | ---: | --- |
| 预训练（dense） | ≈8.76 | ≈1.87 | 初值 = ln(6400)，即均匀猜测 |
| 预训练（MoE） | ≈8.76 | ≈1.96 | 末 100 步均值，σ=0.165 |
| SFT（MoE） | 1.98 | 1.58 | 末 30 步均值 |
| 离线蒸馏（总 loss） | — | 1.189 | `0.5×CE(1.81) + 0.5×KD(0.57)` |
| **OPD（k3 散度）** | **0.2995** | **0.2482** | 首/末 100 步均值，σ=0.0665 |

> **⚠ 不要拿单个 batch 的 loss 下结论**：本项目实测，MoE 预训练最后 100 个采样点里，单点 loss 在 **1.58 到 2.41** 之间摆动（σ=0.165）。我一度根据末尾单点的 1.7075 得出「MoE 击败了 dense 的 1.87」，**随后被 30 点均值 1.9440 推翻**。
>
> **正确做法**：取窗口均值并报标准差。更进一步，训练 loss 不能跨阶段比 —— 要比就在同一批留出数据上重新算。

> **⭐ OPD 那一行是个免费的早停判据。** 19,502 步只把散度从 0.2995 降到 0.2482（−17%），而降幅 0.05 还不到单点波动 σ=0.0665 的一倍 —— **等于说 13.3 小时里几乎什么都没发生**。
>
> 原因很直白：学生和 teacher 都是从同一个 `full_sft` 出发、只差 MoE 与否，KL 意义上本来就很近，**没有多少知识可转移**。这个判断在**前 100 步**就能做出来。详见 §6.5。

## 8.3 效果对比与消融

200 题基准（10 类各 20 题，贪心解码）：

| 模型 | 复读率 | 事实准确率 | 答案为空 | 回复多样性 | 长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pretrain` | 62.7% | **0.0%** | 0.0% | **43.5%** | 530 |
| `full_sft`（基线） | 46.0% | 37.1% | 0.0% | 95.5% | 332 |
| `full_sft_moe`（教师） | 39.1% | 34.3% | 1.0% | 96.5% | 328 |
| `full_dist`（离线蒸馏） | 47.8% | 31.4% | 0.5% | 96.5% | 343 |
| `opd`（在线蒸馏） | 49.4% | 20.0% | 0.5% | 98.5% | 334 |
| `dpo` | 46.3% | 40.0% | 0.0% | 96.5% | 336 |
| `agent`（Agentic RL） | 35.2% | 25.7% | 0.5% | 96.5% | 341 |
| `cispo` | 27.6% | 28.6% | 0.5% | 97.0% | 264 |
| `grpo` | 28.4% | 31.4% | 0.5% | 97.5% | 269 |
| `ppo_actor` | **7.4%** | **0.0%** | **84.0%** | 88.0% | **60** |

> **⚠ 这张表最重要的一行**
>
> **PPO 的复读率 7.4% 是全场最低（看起来最好），事实准确率却是 0.0%** —— 因为它 84% 的回答在 `</think>` 之后是空的。
>
> **只报复读率的排行榜，冠军会是一个根本没有答案的模型。** 抓住它的不是被优化的那个指标，而是「答案为空率」和「回答长度」两个旁证。
>
> 更进一步：官方 PPO 权重**绕过了**这两道守卫（长度 417 全场最长、空答案 0%），只有**相关性**与**准确率**抓住了它。归纳出的规律：**只看输出的指标原则上总能被某种退化绕过；把输出锚定到输入的指标才难被绕过。**

> **SFT 前后的本质差异**：看 `pretrain` 那一行 —— 准确率 0.0%、回复多样性 43.5%、长度 530。它**不是答得差，是根本不在回答**。**SFT 带来的不是「知识」，而是「对话这件事本身」。**

### 留出集困惑度（另一把尺子）

同一批 1600 条留出样本（取自 `dpo.jsonl` 的 chosen，这几个模型都没训练过），teacher forcing 下直接算 loss：

| 模型 | 留出集 PPL |
| --- | ---: |
| `pretrain` | 15.40 |
| `pretrain_moe` | 13.55 |
| `opd` | 12.55 |
| `full_sft`（基线） | 12.26 |
| `full_dist` | **11.91** |
| `full_sft_moe` | **11.20** |

> **⭐ 把这张表和上面那张放在一起看，是全手册最重要的一个对照。**
>
> `full_dist` 的 **PPL 比基线更好**（11.91 < 12.26），**复读率和准确率却都更差**（47.8% vs 46.0%，31.4% vs 37.1%）。
>
> 不矛盾，因为两把尺子量的根本不是一回事：
>
> - **PPL 在 teacher forcing 下测** —— 给完美前文，只问「下一个词猜得准不准」。蒸馏优化的正是这个。
> - **复读率／准确率在自由生成下测** —— 模型自己往下写几百个 token，错误会累积。**这是蒸馏从没训练过的状态（exposure bias，§3.3）。**
>
> **结论：PPL 提升 ≠ 生成质量提升。** 任何只报 PPL 的结论都该被追问一句「自由生成下测了吗」。这也是本项目坚持两套指标一起报的原因。

## 8.4 8GB 显存工程

> **⚠ 最大的坑：显存溢出不报 OOM**
>
> 这台机器的 NVIDIA 驱动开着 **system memory fallback**：显存装不下时**不抛异常**，静默回落到系统内存，速度掉 4–20 倍。
>
> 实测：MoE 预训练 `bs=32` 是 **4.882 s/步**，`bs=16` 是 **0.242 s/步** —— 前者跑完要 215 小时，后者 8.5 小时。**两个都「正常运行」，代价差 25 倍。**

| 撞坑形态 | 触发方式 | 识别信号 |
| --- | --- | --- |
| batch 过大 | 预训练 bs=32 | 4.882 vs 0.242 s/步 |
| 组大小过大 | Agentic RL G=4 | 每样本 6.82 vs 1.72 s（慢 296%，**非线性**） |
| 序列过长 | 默认 `max_total_len=2500` | 单层 scores 763 MB → 真 OOM |
| 外部程序挤占 | 浏览器/IDE 占显存 | GPU 100% 但功耗仅 41 W |

> **可复用的判据**：**「GPU 利用率 100% + 功耗异常低 + 空闲显存不足 200 MB」** 三者同时出现，基本可判定为显存颠簸。真正在算时卡的功耗应接近 TDP。**光看利用率会被骗** —— 等 PCIe 传输也算「忙」。

> **O(S²) 可以先算后验**：朴素注意力单层 scores = `B × H × S² × 4 B`
>
> | 配置 | 单层 scores | 结果 |
> | --- | ---: | --- |
> | `S=2500, B=4` | 763 MB | OOM |
> | `S=1280, B=4` | 200 MB | 能跑但慢 |
> | `S=1280, B=2` | 100 MB | 采用 |
>
> **这三个数是先用公式算出来、再实测确认的。** 面试讲显存优化时，「我先算后验」比「我调小了 batch」高一个层次。

---

# 第九章 · 面试高频题与回答模板

> 每题给出**一句话主线**（加粗，先说结论）+ 展开。建议按主线背，展开部分理解后用自己的话讲。

### Q1 · 请用 2 分钟介绍一下你的 MiniMind 项目。
*项目介绍*

> **我在一张 8GB 的消费级显卡上，从随机初始化开始完整走通了一个 64M 参数大模型的全生命周期：分词器、预训练、SFT、知识蒸馏、四条对齐路线，一共 86 GPU 小时、六个阶段零崩溃。**

**结构层面**是 Decoder-only：8 层、hidden 768、GQA 8 查询头配 4 个 KV 头、RoPE base 取 1e6、RMSNorm、SwiGLU，另外做了 MoE 版本（4 专家 top-1，198M 总参但激活只有 63.94M，与 dense 的 63.91M 几乎相同，所以两者可以公平对比）。

**但这个项目我最想讲的不是跑通了多少阶段，而是评测。** 我发现只用复读率这一个指标时，排行榜冠亚军是两个根本没有答案的坏模型 —— PPO 有 84% 的回答是空的，却拿了最低的复读率。为此我建了 200 题基准、加了准确率、相关性、答案为空率等六道正交守卫，还用多种子重训量化了训练方差，据此**主动撤回了自己之前写下的四条结论**。

**收尾**：这个项目让我具体地知道了，一个漂亮的指标在什么情况下是假的，以及该用什么去交叉验证它。

---

### Q2 · 为什么用 GQA 而不是 MHA？减的为什么是 KV 而不是 Q？
*架构细节*

> **因为推理时被缓存的只有 K 和 V，Q 每步新算完就丢，所以要压显存就只能压 KV。**

MiniMind 用 8 个 Q 头配 4 个 KV 头，`n_rep=2`，KV Cache 直接减半。代码里就是 `repeat_kv(xk, 2)`。再往下是 MQA（所有 Q 头共用 1 份 KV），缓存降到 1/8 但质量损失明显，**GQA 是这条线上的折中**。

**量化一下**：`2 × 8层 × 4头 × 96 × 2B = 12 KB/token`，MHA 则是 24 KB。长上下文推理时这直接决定能开多大并发。

---

### Q3 · RoPE 怎么把绝对位置变成相对位置？为什么能外推？
*架构细节*

> **RoPE 不给向量「加」位置信息，而是按位置把 Q、K 旋转一个角度；两个向量做内积时，结果只依赖它们的位置之差。**

数学上 `⟨R(m)q, R(n)k⟩ = f(q, k, m−n)`。**直觉是两根时钟指针 —— 注意力算的是夹角，夹角只跟「差几格」有关。**

**关于外推**要说清楚 RoPE 本身**并不天然外推**：超出训练长度会遇到没见过的角度，效果崩。真正让它能外推的是两件事：① **把 base 调大**（MiniMind 用 1e6 而非 1e4，低频周期从 6.3 万拉到 628 万）；② **YaRN 分段插值** —— 高频不动、低频除以 factor 压回训练范围、中间 ramp 过渡，MiniMind 做成推理期开关不用重训。

---

### Q4 · 为什么交叉熵就是语言模型的正确损失？
*训练机制 · 地基题*

> **因为交叉熵就是最大似然本身 —— 不是「随便选的一个损失函数」。**

推导四步：① 语言模型要最大化语料的概率 `P(x₁…x_T)`；② 链式法则把它拆成 `∏ P(x_t|x_<t)`，这一步是恒等变形没有近似；③ 连乘会下溢，取对数变连加；④ 加负号变成最小化，得到**负对数似然 NLL**。

而 NLL **恰好等于**交叉熵 —— 因为真实分布是 one-hot，交叉熵 `−Σ q log p` 里只有真值那一项非零。

**再补一个能立刻用上的推论**：随机初始化时输出接近均匀分布，所以 loss ≈ `ln(vocab_size)`。MiniMind 是 ln(6400)=8.76。**这是排查训练脚本最快的第一个检查点** —— 远大于说明初始化或 label 有问题，远小于说明标签泄漏（最常见是忘了 shift）。

---

### Q5 · SFT 时如何只对 Answer 计算 loss，Mask 掉 Prompt？
*训练机制 · 最高频*

> **把 labels 初始化为全 −100，再用 token id 序列匹配定位每一段 assistant 回答，只把这些区间填回真实 id；`cross_entropy(ignore_index=-100)` 会自动跳过其余位置。**

MiniMind 的实现在 `SFTDataset.generate_labels`：以 `<|im_start|>assistant\n` 的 token 序列为起点标记，扫到 `<|im_end|>` 为终点，中间全部填回。

**要强调一点**：prompt 部分**依然完整参与前向、依然被注意力看到**，只是不产生梯度。模型学的是「给定这个 prompt 该回什么」，不是「怎么把 prompt 写出来」。

**三个能拉开差距的细节**：① 多轮对话有**多个** assistant 段，必须循环扫完；② `<|im_end|>` 本身**要计入** loss，否则模型学不会停；③ 匹配必须在 **token id 层面** —— 同样的文字在不同上下文可能切成不同 token。

---

### Q6 · 什么是 Teacher Forcing？它有什么副作用？
*训练机制*

> **训练时不管模型第 t 步预测成什么，第 t+1 步喂进去的都是真实的第 t 个词 —— 这样 T 个位置可以并行算完，而不用像推理那样串行 T 次。**

**副作用是 exposure bias**：训练时模型看到的永远是完美前文，推理时看到的是自己生成的、可能有错的前文。**一旦第一步错了，后面就在训练中从没见过的分布上走。**

**这正是 RL 和 on-policy 蒸馏存在的根本理由** —— 让模型在自己会走到的状态上被评价和修正，而不是只在别人写好的正确轨迹上学。能把 Teacher Forcing 和 RL 的必要性串起来讲，说明理解到位了。

---

### Q7 · 策略梯度是什么？为什么需要基线？
*强化学习 · 地基题*

> **策略梯度定理说：`∇J = E[ R · ∇log π ]` —— 拿到高分的回答就提高它的对数概率，低分的就压低，R 就是每个样本梯度的权重。**

关键推导是**对数导数技巧** `∇π = π·∇log π`，它把「对分布求导」变回了「在分布下求期望」，于是可以用采样来估计。

**为什么需要基线**：原始形式方差极大。假设所有回答分数都在 5 到 7 之间 —— 它们全是正的，于是**所有**回答的概率都被推高，真正有用的信号（「7 分比 5 分好」）淹没在共同偏移里。

减去一个不依赖动作的基线 b，**期望不变**（因为 `E[∇log π] = ∇(Σπ) = ∇1 = 0`）**但方差大幅下降** —— 这是免费的午餐。于是 `A = R − b`，问的从「好不好」变成「**比平均好多少**」，也就有了真正的负信号。

---

### Q8 · PPO 的裁剪为什么外面要套一个 min？
*强化学习 · 高区分度*

> **因为如果只做 clip，当 ratio 已经超出范围时梯度就恒为 0，模型「跑错方向跑太远」之后再也回不来了。min 保证了这种情况下梯度仍然起作用。**

`L = min( ratio·A , clip(ratio, 1−ε, 1+ε)·A )`。分四种情况看：

① **好动作已提升很多**（A>0, ratio>1+ε）→ min 选 clip 项 → 梯度截断，不再继续推高。
② **坏动作已压低很多**（A<0, ratio<1−ε）→ min 选 clip 项 → 截断。
③ 正常范围 → 选 ratio 项 → 正常更新。
④ **坏动作反而被推高了**（A<0, ratio>1+ε）→ 此时 ratio 项**更负**，min 选中它 → **不截断，让梯度把它拉回来。**

**第四种情况就是 min 的全部意义**：裁剪只在「已经朝对的方向走够了」时刹车，绝不在「走错方向」时刹车。

---

### Q9 · GAE 是干什么的？为什么需要它？
*强化学习*

> **GAE 把「整段回答的一个分数」合理地分摊回每一个 token，让每个 token 都有自己的优势值。**

先是 **TD 误差** `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)`，直觉是「我原本以为这局值 V(s_t) 分，走一步后实际拿到 r_t 且新局面值 V(s_{t+1}) 分，**δ 就是这次比预期好了多少**」。

然后 **GAE 把未来所有惊喜按 γλ 衰减加起来**：`A_t = δ_t + γλ·A_{t+1}`，倒着递推一遍算完。λ 控制偏差-方差权衡：λ=0 只看一步（偏差大方差小），λ=1 看完整轨迹（无偏方差大）。本仓库用 `λ=0.95, γ=1.0`。

**为什么在 LLM 里特别重要**：本仓库的奖励是**稀疏终局奖励** —— 中间 token 全是 0，整段回答的分数只加在最后一个 token 上。**GAE 的作用就是把这个终局分数分摊回前面每一个 token**，否则前面的 token 拿不到任何学习信号。

---

### Q10 · GRPO 为什么能省掉 Critic？代价是什么？
*强化学习 · 算法对比*

> **PPO 用 Critic 估计状态价值来当基线；GRPO 换成「同一个 prompt 采 G 个回答，用这一组的均值当基线」。**

具体是 `A⁽ⁱ⁾ = (r⁽ⁱ⁾ − mean(组)) / (std(组) + 1e-4)`。这 G 个回答面对**同一个 prompt**、难度完全一样，所以它们的平均分天然就是好基线。除以标准差是为了让不同难度的 prompt 产生的优势**尺度一致**。

**省掉一整个价值网络**，显存和训练成本都降一大截。

**代价是组内样本数 G 太小时基线噪声很大。** 我在 Agentic RL 阶段因显存所限被迫用 G=2，日志里的 `GrpStd` 抖动明显 —— 这条结果不应和 G=6 的并排当同等口径比较。**这类方法学代价必须主动标注。**

---

### Q11 · CISPO 和 GRPO 差在哪？
*强化学习 · 源码级*

> **代码上只差一行，但形式完全不同：GRPO 是 PPO 式的（ratio 带梯度、参与优化目标），CISPO 是 REINFORCE 式的（ratio 被 detach，只当权重系数）。**

`GRPO : L = −min(ratio·A, clip(ratio)·A)`
`CISPO: L = −clamp(ratio).detach() · A · log π`

**CISPO 回到了最原始的 `A·∇log π` 形式**，只是给它乘一个被切断梯度的重要性权重作修正。detach 之后 ratio 不参与反向，梯度形式更简单更稳定。

另外 `clamp(max=ε_high)` **只截上界**（本仓库 5.0），防止个别样本权重过大主导梯度，**不截下界** —— 低概率样本权重小本来就不危险。

**本项目实测两者收益相当**：复读率 27.6% vs 28.4%，差 0.75pp。但这个差异**小于训练噪声**（多种子实测 σ≈0.80pp），所以正确表述是「未能区分」而非「确认无差异」。

---

### Q12 · DPO 是怎么把奖励模型消掉的？
*强化学习 · 推导题*

> **因为 RLHF 那个「最大化奖励 + KL 约束」的优化问题有闭式解，反解出来发现奖励可以用策略表示；再代入 Bradley-Terry 模型时，那个讨厌的配分函数在<u>相减</u>时被完全消掉。**

五步推导：① 目标 `max E[r] − β·KL(π‖π_ref)`；② 闭式解 `π* ∝ π_ref · exp(r/β)`；③ 反解 `r = β·log(π*/π_ref) + β·log Z(x)`；④ 代入 BT 模型 `P(y_w≻y_l) = σ(r_w − r_l)`，**log Z(x) 只依赖 x，相减时抵消**；⑤ 最大似然得到 DPO 损失。

`L = −log σ( β·[(logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l))] )`

**为什么必须减 π_ref**：只看 π_θ 的话，模型可以把 chosen 和 rejected 的概率**一起压低**来降 loss —— 那是灾难性遗忘。减去参考项后只有**相对**变化才算数。

**代价**：DPO 受限于成对数据的覆盖范围，无法像 PPO 那样通过采样探索超出数据分布的策略。

---

### Q13 · DPO 里的 β 有什么作用？
*强化学习*

> **β 控制策略允许偏离参考模型多远，等价于 KL 约束强度的倒数。**

**β 小（0.01）**：约束松，学得快，但容易过拟合偏好数据、丢通用能力，严重时胡言乱语。**β 大（0.5）**：约束紧，贴着参考模型，稳但学不动。**常用 0.1**，本仓库默认也是 0.1。

**从公式看**：β 是 logsigmoid 输入的缩放因子。β 越大，同样的 logratio 差距越快进入 sigmoid 饱和区，梯度越小、更新越保守。

**补一句实测**：我在 64M 规模跑 DPO 发现完全无效 —— 权重相对变化只有 0.0055%，低于 fp16 存储精度 0.098%，等于什么都没改。两个独立指标（复读率、奖励模型打分）一致确认无变化。**这类负结果比多报一个正结果更能说明会做验证。**

---

### Q14 · KL 惩罚为什么用 `exp(r)−r−1` 而不是直接 `−r`？
*强化学习 · 源码级*

> **因为那是 k3 估计量 —— 它同时做到了恒非负、无偏、低方差，而简单的 k1 做不到。**

记 `r = log π_ref − log π_θ`，三种估计量：
`k1 = −r`：无偏但方差大，**而且可能为负**（KL 本不该为负）。
`k2 = r²/2`：恒非负、方差小，但**有偏**。
`k3 = exp(r) − r − 1`：**三者兼得**。

**为什么 k3 恒非负**：`e^r ≥ 1+r` 对所有实数成立（指数函数在 r=0 处的切线），所以 `e^r − r − 1 ≥ 0`，等号仅在两分布相同时取到。这是 John Schulman 提出的，现在是事实标准。

**本仓库两处细节值得说**：① KL 是**加在 loss 里**而非加在 reward 里（经典 RLHF 是后者，会经 GAE 分摊到每个 token）；② **早停判据用的是 k2**（`0.5·log_ratio²`，阈值 0.25），因为早停只需要一个偏离程度的标量，k2 不用算 exp 更省。

---

### Q15 · LoRA 的 B 为什么初始化为 0？rank 怎么选？
*微调 · 高频*

> **B=0 保证训练起点 ΔW = B·A = 0，模型行为与原模型完全一致，不会因为随机适配器扰动而在初期崩坏。**

**为什么不能都随机**：起点就带了一个随机扰动，等于给训练好的模型加噪声。**为什么不能都置零**：`∂L/∂A = Bᵀ(...) = 0`，梯度恒为零永远学不动。**所以必须一个随机一个置零，且置零的要是输出侧的 B。**

**rank 怎么选**：任务与预训练分布越远、要注入的新能力越多，rank 越大。风格适配 r=4~8 够，领域知识注入常用 16~64。

**还有一个常见误解要澄清**：LoRA 省的是「梯度存储 + 优化器状态」，**不省激活值** —— 反向传播依然要穿过整个网络才能算出上游梯度。要省激活得用梯度检查点。AdamW 每个可训练参数要存两个 fp32 动量，全量微调是 511 MB，LoRA 只要 3.1 MB，**这才是省显存的大头**。

---

### Q16 · 你读过 MiniMind 的 LoRA 实现吗？有什么问题？
*微调 · 源码级 · 极高区分度*

> **有两处与标准 LoRA 的偏离：只给方阵挂适配器，以及没有 alpha 缩放。**

**偏离一**：`apply_lora` 的筛选条件是 `in_features == out_features`。在 MiniMind 里只有 `q_proj` 和 `o_proj` 是 768×768；**k_proj/v_proj 因为 GQA 变成 768×384、FFN 是 768×2432，全都挂不上**。所以实际只有 16 个模块、393,216 参数、占 0.62%（与日志吻合）。

而 **LoRA 原论文的消融结论是「只改 W_q 和 W_v 效果就很好」** —— 这里因为 GQA 漏掉了 v_proj，实际改的是 W_q 和 W_o，**与论文推荐并不一致**。

**偏离二**：`forward` 是 `return self.B(self.A(x))`，**没有标准 LoRA 的 α/r 缩放**。α/r 的作用是让**换 rank 时不必重调学习率** —— r 变大时 B·A 的典型幅度也变大，除以 r 正好抵消。**没有它，把 r 从 8 改到 64 时等效更新幅度会跟着变，学习率必须重调。** 对固定 r 的单次实验无影响，但做 rank 消融会得到被混淆的结论。

---

### Q17 · 训练出现 Loss Spike 或 NaN，怎么排查？
*工程坑点 · 高频*

> **按「先定位是数据、还是数值、还是优化」的顺序查，从最便宜的检查做起。**

**第一步：看开局 loss 对不对。** 应该 ≈ `ln(vocab_size)`，MiniMind 是 8.76。远大于说明初始化或 label 有问题；**远小于说明标签泄漏** —— 最常见是忘了 shift。

**第二步：定位到具体 batch。** 固定种子复现，dump 爆炸前几步的数据。常见元凶是超长样本、全是重复字符的脏数据、或某条样本 label 全是 −100（该 batch 的 loss 变成 0/0）。

**第三步：数值层面。** fp16 动态范围窄 → 优先换 **bf16**；检查归一化层是否在 fp32 下计算；确认 `scaler.unscale_` 在 `clip_grad_norm_` **之前**调用（顺序反了裁剪就没意义）。

**第四步：优化层面。** 降 lr、加 warmup、收紧 grad_clip。

**结构层面的预防**：MiniMind 在 Q、K 上各挂了一个 RMSNorm（**QK-Norm**），专门把注意力 logits 尺度钉住，避免 softmax 饱和引发 spike。加上 Pre-Norm 的干净残差通路，这类问题在这个规模基本不出现。

---

### Q18 · 显存 OOM 怎么优化？按什么顺序试？
*工程坑点 · 高频*

> **按「收益/代价」排序：先调不损失效果的（累积、精度、序列长度），再调有代价的（重计算、卸载）。**

① **梯度累积**：batch 减半、累积翻倍，等效批量不变、数学等价，几乎零代价。② **bf16**：激活显存减半。③ **缩短序列长度**：注意力显存随 S² 增长，收益最大。④ **梯度检查点**：约 30% 额外计算换掉大部分激活显存。⑤ **优化器状态卸载 / 8-bit optimizer**：AdamW 状态是权重的两倍。⑥ **LoRA / QLoRA**。

**但我想强调一个更前置的问题**：在我这台机器上，**显存溢出根本不报 OOM** —— 驱动开着 system memory fallback，装不下时静默回落到内存，速度掉 4 到 20 倍。实测 bs=32 是 4.882 s/步、bs=16 是 0.242 s/步，**两个都「正常运行」，跑完时间差 25 倍。**

**识别判据**：GPU 利用率 100% + 功耗异常低（我实测 41W）+ 空闲显存不足 200MB。**光看利用率会被骗，因为等 PCIe 传输也算「忙」。** 所以我写了个显存探针：真建模型、真跑 5 步前反向、读 `max_memory_reserved()`，按实测选 batch。

---

### Q19 · 开了 Flash Attention 为什么还会 OOM？
*工程坑点 · 源码级*

> **因为 SDPA 的快速路径有前提条件，条件不满足时会静默回落到朴素实现，而朴素实现的显存是 O(S²)。**

MiniMind 的条件是 `if self.flash and (seq_len>1) and (past_key_value is None) and (mask is None or all(mask==1))`。**只要带了 KV Cache，或 attention_mask 里有 0（有 padding），就走 else 分支**，实体化 `[B,H,S,S]` 的 scores。

**我实测过**：多轮工具调用累积到 S=2500、组大小 4 时单层 scores 就是 `4×8×2500²×4B = 763 MB`，8 层加反向直接 OOM。降到 S=1280、组大小 2 后是 100 MB 才跑得动。

**加分点**：这三个配置的显存我是**先用 `B×H×S²×4` 算出来、再实测确认的**。而且发现组大小从 4 降到 2 时**每样本耗时快了 296% 而不是 100%** —— O(S²) 注意力与显存回落是复合效应，长序列任务上 `num_generations` 根本不是线性成本参数。

---

### Q20 · MoE 的 aux_loss 是干什么的？怎么证明专家没坍缩？
*架构细节*

> **aux_loss 防的是「路由器把所有 token 都送给同一个专家」—— 那样 MoE 会退化成 dense，白占几倍显存。**

实现是 `Σ(实际命中率 × 平均路由概率) × num_experts × 5e-4`，分布均匀时取最小。**它同时惩罚「命中多」和「打分高」**，所以路由器没法靠只提高分数而不实际路由来钻空子。

**但只看 aux_loss 稳定是间接证据** —— 它是标量，稳定只说明损失没恶化。**直接做法**是在 `gate` 上挂前向钩子，取出路由 logits、按模型自身口径复原 top-k 分配，逐层统计。

我实测：4 个专家占比 **25.9 / 25.0 / 24.9 / 24.3%**，归一化熵 1.000，8 层全均匀，零个未使用专家。**给出这组数字，比说「aux_loss 很稳」强一个量级。**

---

### Q21 · 为什么 MoE 总参 198M 却说和 64M 的 dense 可比？
*架构细节*

> **因为 top-1 路由下每个 token 只走 1 个专家，实际激活参数是 63.94M，与 dense 的 63.91M 几乎相同 —— 单步前向的计算量可比。**

MoE 每层把 dense 的 MLP（5.6M）换成 gate（3072）+ 4 个专家（4×5.6M），总参涨到 198.42M；但 `num_experts_per_tok=1` 意味着每 token 只激活一个专家。`激活 = 8×(注意力1.77M + 单专家5.6M + gate) + embed 4.92M = 63.94M`。

**不公平的地方要主动说**：全部专家都要常驻显存，MoE 权重文件 407MB 而 dense 只有 131MB；推理延迟实测 3.28 s/题 vs 2.0 s/题（专家路由的 scatter/gather 有开销）。

**准确表述**：在同等激活计算量下，MoE 用 3 倍显存换来了留出集困惑度 12% 的下降。

---

### Q22 · 你怎么判断一个评测指标是不是被「刷」了？
*方法学 · 区分度最高*

> **看这个指标有没有「锚定到输入」。只审视输出长什么样的指标，原则上总能被某种退化绕过。**

我抓到过两个「指标很好但模型是坏的」的例子，坏法完全不同：

- **例一**：我的 PPO 复读率 7.4% 全场最低，但 84% 的回答在 `</think>` 之后是空的，准确率 0.0%。**抓住它的是「回答长度」和「答案为空率」。**
- **例二**：官方 PPO 权重复读率 12.2% 全场第二，长度 417 全场最长、空答案率 0% —— **把上面两道守卫全绕过了**。但它 200 道题只产出 44 种不同开头，无论问什么都回同一篇 AI 伦理散文。

**归纳的规律**：复读率、长度、空答案率、多样性全都只看输出的形状；相关性（答案有没有提到问题里被比较的两个对象）和准确率则把输出**与输入对照**。后两个才难被绕过。

**再补一条更重要的**：统计显著性保护不了你。我那个「复读率降低 78.9%、p<0.0001」是完全真实的测量、完全错误的解读。**p 值只保证你没被随机性骗到，不保证你量对了东西。**

---

### Q23 · 你怎么确定实验结论不是运气？
*方法学*

> **要分别量化两种不确定性 —— 评测噪声和训练噪声，而大多数人只量了前一半。**

**评测噪声**：换一批题目考，分数怎么波动。用**配对自助法**（同一批题上比较，重采样 10000 次）。配对能消掉题目难度带来的方差，比独立比较敏感一个量级。

**训练噪声**：同配置只换随机种子重训一遍，模型会差多少。**这一半几乎没人量，因为很多框架根本没给做重复实验的接口** —— MiniMind 的种子就是硬编码 42 的，我加了 `--seed` 参数才做得了。

**实测有个我完全没预料到的发现**：训练方差是**指标的属性**，不是模型的属性。同一批权重，复读率的种子间标准差只有 **0.55pp**，事实准确率却有 **4.36pp** —— 相差 8 倍。**所以不能从一个指标外推到另一个。**

**拿它重新定级**：核心结论（策略优化降复读 18pp）在保守上界下仍有 9.3 倍标准差，稳；两条小效应结论只有 1–2 倍，**我据此撤回了它们的显著性表述**。

---

### Q25 · 知识蒸馏为什么比直接用标签训练更有效？
*蒸馏 · 高频基础题*

> **因为 one-hot 标签每个位置只给 1 个数的监督，teacher 分布给的是整个词表 6400 个数 —— 同样的数据，梯度里携带的信息密度差三个数量级。**

标签只说「这里该是『很』」。teacher 还顺带告诉学生「『非常』也挺合适、『但是』完全不行」。这些**错误选项上的相对分数**就是所谓的**暗知识（dark knowledge）**，它编码了 teacher 对词与词之间相似关系的理解。

两个损失的形式其实完全一样，只是把 one-hot 的 `q` 换成了 teacher 的软分布：

```
SFT  ：L = − Σ_v  q(v) · log p_student(v)           q 是 one-hot
蒸馏 ：L = − Σ_v  p_teacher(v) · log p_student(v)    6400 项全非零
```

代码里用 `F.kl_div` 而不是交叉熵，差的只是一个与学生无关的常数（teacher 的熵），对梯度没有任何影响。

**加分点**：本仓库用的是混合损失 `alpha*CE + (1-alpha)*KD`，`alpha=0.5`。**保留 CE 项的意义是 teacher 也会错** —— 只学 teacher，学生的上限被钉死在 teacher 上，且会原样继承它的错误。

---

### Q26 · 蒸馏温度 T 是干什么的？那个 T² 为什么必须乘回去？
*蒸馏 · 能拉开差距的细节*

> **T 把 teacher 的分布「压平」，让藏在小数点后几位的暗知识显现出来；T² 是为了补偿升温带来的梯度衰减。**

训练好的 teacher 往往极度自信 —— 正确词概率可能 0.999，其余 6399 个词加起来才 0.001，暗知识全挤在小数点后好几位，学生几乎看不见。`softmax(z/T)` 中 T>1 就是把这些差异放大到可感知的范围（T→0 退化成 one-hot，T→∞ 变均匀分布、信息全丢）。

**T² 的来历**：对学生 logits 求导时，`z/T` 这一层带出一个 `1/T`；teacher 侧的软化又贡献一个 `1/T`。**净效果是梯度被缩小到 1/T²。**

**不补偿的后果**：`T=1.5` 时蒸馏项梯度只有 CE 项的 1/2.25，你以为在做 50:50 的混合，**实际是 69:31** —— `alpha` 这个超参失去了意义，而且换 T 就得重调学习率。

```python
return (temperature ** 2) * kl      # trainer/train_distillation.py L36
```

---

### Q27 · 什么是 On-Policy Distillation？它和普通蒸馏、和 RL 分别差在哪？
*蒸馏 · 进阶题，能答好说明真读懂了 exposure bias*

> **一句话：OPD = 强化学习的「在线采样」结构 + 知识蒸馏的「稠密信号」，而且完全不需要奖励模型。**

用比方说清楚区别：**离线蒸馏是老师批改课本上的范文；OPD 是学生自己写一篇，老师在学生自己写的每一句上批注。差别只有一个 —— 前文是谁写的。**

```
离线蒸馏： L = E                [ Σ  D(π ‖ ν) ]     ← y 来自数据集
               x,y ~ 数据集        t    θ

OPD    ： L = E                [ (1/|y|) Σ  D(π ‖ ν) ]   ← y 来自学生自己
               x~数据, y~π (·|x)          t    θ
                          θ
```

**为什么这很重要**：离线蒸馏依然是 teacher forcing（§3.3），喂的永远是完美前文。**学生在自己会犯的错误之后该怎么补救，teacher 一次都没教过。** OPD 把采样换成学生自己的，正好补上这个缺口。

和 RL 比，结构完全一样（都是 on-policy 采样），**差别在信号**：

| | 前文来自 | 信号 | 信号密度 | 要奖励模型吗 |
| --- | --- | --- | --- | --- |
| SFT | 数据集 | one-hot | 1 数/token | 不要 |
| 离线蒸馏 | 数据集 | teacher 分布 | 6400 数/token | 不要 |
| RL（GRPO 等） | **学生自己** | 标量奖励 | **1 数/整条序列** | **要** |
| **OPD** | **学生自己** | **teacher 分布** | **6400 数/token** | **不要** |

OPD 占了最好的那一格。**代价是必须有一个真的比自己强的 teacher** —— 这正是本项目栽跟头的地方（见 Q29）。

**实现细节加分点**：本仓库默认 `loss_mode=k3`，只在**采样到的那一个词**上算 `exp(r)−r−1`，而不是全词表 KL。这是被 8GB 显存逼出来的选择（全词表要实体化两份 `[B,R,6400]`），但恰好也是 §5.7 里方差最小的那个估计量。

---

### Q28 · Agentic RL 和普通 RL 的结构性区别是什么？
*Agentic · 核心必答题*

> **只有一个：上下文里出现了不是模型生成的 token（工具返回的结果），这些位置必须屏蔽梯度。**

损失函数、优势估计、KL 惩罚，全部原封不动复用 GRPO / CISPO。区别就在 mask：

```python
response_mask.extend([1] * len(new_ids))     # 模型生成的 → 有梯度
...执行工具...
response_mask.extend([0] * len(obs_delta))   # 工具返回的 → 无梯度
```

**为什么必须屏蔽**（两条理由都要说）：

1. **会教错东西。** 策略梯度 `∇J = E[A·∇log π(动作)]` 里的「动作」必须是策略自己选的。工具返回的 `391` 是**环境给的观测**，对它算梯度等于在教模型「预测计算器会输出什么」—— 我们要的是它学会**什么时候调工具**，不是**背下工具的答案**。
2. **会稀释信号。** 工具输出是确定性的，模型很快背下来，`log π` 趋近 0，这些位置贡献的梯度极小却占着归一化的分母，把真正的动作 token 的信号压下去。

**再加两个工程细节**：① 工具结果要截断（`[:2048]`）—— 模型写出 `9**9**9` 会得到几十万位的数字，撑爆 tokenizer；② `eval` 必须清空 `__builtins__` 并加超时，否则模型生成的表达式可以执行任意代码。

---

### Q29 · 你的蒸馏和 OPD 实验都失败了，怎么解释？
*收尾 · 考察诚实度与归因能力*

> **失败的原因很明确：teacher 本身不够强。teacher 的准确率 34.3%，比学生起点的 37.1% 还低 —— 蒸馏的上限是 teacher，让强学生去对齐弱老师，结果向下收敛完全符合预期。**

三条数据支撑这个归因：

1. **OPD 比离线蒸馏掉得更多**（准确率 20.0% vs 31.4%）。这**反而验证了 OPD 的机制是对的** —— 信号更稠密、又直接施加在学生自己的分布上，所以**向 teacher 收敛得更彻底**。方法没问题，是目标选错了。
2. **训练日志本身就预告了结果**：19,502 步散度只从 0.2995 降到 0.2482（−17%），降幅 0.05 还不到单点波动 σ=0.0665 的一倍。学生和 teacher 都从同一个 `full_sft` 出发、只差 MoE 与否，**KL 意义上本来就很近，没有多少知识可转移**。
3. **我选 teacher 时用错了指标**。teacher 的留出集 PPL 是全场最低的 11.20，看着最强 —— 但 PPL 是 teacher forcing 下测的，生成质量是自由生成下测的。`full_dist` 就是最好的反例：**PPL 比基线好（11.91 < 12.26），复读率和准确率却都更差。**

**两条能带走的方法论**：

- **选 teacher 要用你最终关心的那个指标，不能用代理指标。** 如果当初用「200 题准确率」去挑，会发现根本没有比 `full_sft` 更合适的老师 —— 那么正确的决策是**不做这个实验**。
- **蒸馏开跑后第一件要看的事是散度降不降。** 不降 = 没知识可转移，这个判断**前 100 步就能做出来，不用等 13 小时**。

> 这两条是花 13.3 GPU 小时买来的。**面试时主动讲一个失败实验并把归因讲清楚，比讲三个成功实验更有说服力** —— 成功可能是运气，能准确解释失败才说明你控制得住这套方法。

---

### Q24 · 这个项目的局限是什么？你会怎么改进？
*收尾 · 考察诚实度*

> **最大的局限是没有人工或强模型评判 —— 我所有指标都是程序化的，复读率只是生成质量的粗糙代理。**

**其余几条**：⓪ **蒸馏与 OPD 的 teacher 选错了** —— 我用留出集 PPL 挑 teacher，而 PPL 低不代表生成质量好，13.3+4.3 GPU 小时没换来任何提升（详见 Q29）。① **方差估计只有 n=2/n=3**，点估计可信但上界很宽（0.3–1.9pp）。② **所有方法结论只在数据受限体制下成立** —— 官方全量数据模型比 mini 数据好 29.8pp，超过我全部方法收益之和，换到数据充足的设定，方法之间的相对关系可能完全不同。③ **单语言、单领域**，全是中文通用对话。④ 模型本身很弱，事实准确率只有 37% —— 这是 64M 参数的固有限制。

**如果有更多资源，优先级是**：先把评测做实（加 LLM-as-judge、扩带格式约束的题目），而不是把模型加大。**在评测工具还查不出 5pp 差异的时候加大模型，只会得到更多「不显著」。**

---

## 附：一页速查

| 要点 | 一句话 |
| --- | --- |
| 起点 loss | ≈ ln(vocab) = ln(6400) = **8.76**，偏离说明有 bug |
| 交叉熵 = NLL | 不是随便选的损失，它就是最大似然本身 |
| shift | `logits[:-1]` 对 `labels[1:]`，忘了就是预测自己 |
| Teacher Forcing | 并行训练的代价是 exposure bias，这是 RL 存在的理由 |
| SFT mask | labels 全 −100，只填回 assistant 段（含 im_end） |
| GQA | 压 KV 不压 Q，因为只有 KV 进缓存；n_rep = 8/4 = 2 |
| QK-Norm | Q/K 各挂 RMSNorm，在 RoPE 之前，防 logits 爆炸 |
| RoPE | 内积只依赖 m−n；base=1e6 拉长低频周期；YaRN 推理期外推 |
| RMSNorm | 不减均值无偏置；内部转 fp32 再降回 |
| SwiGLU | `down(SiLU(gate)⊙up)`，3 个矩阵，中间维 2432 ≈ 3.17× |
| tie_embeddings | 省 4.92M（7.7%）；统计参数量时只算一次 |
| MoE | 198.42M 总参 / 63.94M 激活；aux_loss 系数 5e-4 |
| LR 调度 | 余弦从 1.0 衰减到 **0.1 不到 0**，且**无 warmup** |
| 累积顺序 | 先除 accum → backward → `unscale_` → clip → step |
| 策略梯度 | `∇J = E[R·∇log π]`；减基线不改期望但降方差 |
| PPO 的 min | 只在「走对方向走够了」时刹车，绝不在走错时刹车 |
| GAE | 把稀疏终局奖励分摊回每个 token；λ 控偏差-方差 |
| KL 的 k3 | `exp(r)−r−1`：恒非负 + 无偏 + 低方差，三者兼得 |
| GRPO | 组内均值当基线，省掉 Critic；G 小则基线噪声大 |
| CISPO | ratio 被 detach，只当权重 → REINFORCE 式而非 PPO 式 |
| DPO | β=0.1 控偏离；减 π_ref 防两边一起压低；配分函数在相减时消掉 |
| LoRA | B 初始化为 0；省的是优化器状态不是激活；本仓库无 α 缩放且只挂方阵 |
| Flash 回落 | 带 KV Cache 或 mask 有 0 → 走朴素路径，显存 O(S²) |
| sysmem fallback | 不报 OOM 只降速；判据 = 100% 利用率 + 低功耗 + 显存贴顶 |
| 指标可信度 | 只看输出的指标会被绕过；要有锚定输入的指标 |
| 蒸馏 | 学 teacher 的整张分布（6400 数/token）而非 one-hot（1 数/token） |
| 蒸馏温度 | T 压平分布放大暗知识；乘 T² 补偿被缩小的梯度 |
| OPD | RL 的在线采样 + 蒸馏的稠密信号，且**不要奖励模型**；上限是 teacher |
| Agentic RL | 与 GRPO 同一个 loss，唯一区别是**工具返回的 token 必须 mask 掉梯度** |
| PPL vs 生成 | PPL 在 teacher forcing 下测、生成质量在自由生成下测，**可以反向动** |
| 选 teacher | 用你最终关心的指标选，别用 PPL 这种代理指标 |
| 两种噪声 | 评测噪声（配对自助法）+ 训练噪声（多种子重训） |

---

## 相关脚本

本手册第八章引用的所有实测数据，都可以用仓库里的评测套件复现：

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
