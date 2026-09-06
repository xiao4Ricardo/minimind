# evals — 训练成果的评测与验收脚本

报告里的每个数字都由这里的脚本产出，全部可当场重跑复现。所有脚本从仓库任意位置调用皆可
（内部自行定位仓库根目录），依赖 `out/` 下的权重与 `dataset/` 下的数据。

## 为什么这些脚本长这样

三条贯穿始终的方法学，都是踩坑换来的：

**一、单点数值不能下结论，必须取窗口均值。**
训练日志里的 "final loss" 是单个 batch 的值。MoE 预训练最后 100 个采样点中，单点在
1.58 ~ 2.41 之间摆动（σ=0.165）—— 拿两次训练各自的最后一个点比大小是没有意义的。

**二、比较必须配对，且各模型的输入必须逐字节相同。**
`SFTDataset` 取样时会随机决定是否加 system prompt、是否删除空 think 标签。给每个模型
各迭代一次 DataLoader，各模型看到的其实是不同版本的数据，"配对"就是假的。
`paired_ppl_test.py` 先把 batch 固化成一份张量再喂给所有模型，并校验各模型的 token
计数完全一致。修正后置信区间收窄约一个量级。

**三、主指标再显著也可能量错了东西，必须有未被优化的旁证指标。**
本项目抓到过两例"因为什么都不说而在主指标上夺冠"的模型：
- PPO：奖励 +0.69 全场最高、20 题复读率 9.6% 全场最低，两项都 p<0.001 —— 但 91% 的
  采样输出在 `</think>` 之后为空。抓住它的是回答长度（94 vs 其余 ~300）。
- pretrain_moe：200 题复读率 16.8% 排全场第二 —— 但 67% 的回答是 0 字符的空串。
  抓住它的是新增的退化率（67.0%）与中文占比（1.9%）。

## 脚本

| 脚本 | 作用 |
| --- | --- |
| `eval_bench200.py` | 200 题 / 10 类生成基准，贪心解码，逐模型产出原始回复到 `bench200_raw.json` |
| `analyze_bench200.py` | 对上述原始输出做统计：配对自助法 + 符号检验、新旧基准交叉验证、分类别拆解 |
| `paired_ppl_test.py` | 留出集 loss/PPL 的配对显著性检验（固化 batch，校验 token 计数一致） |
| `reward_curve_rebuild.py` | 从检查点重建奖励曲线，含 GRPO 校准对照 |
| `reward_decompose.py` | 把奖励按公式逐项拆开（规则项 / 复读惩罚 / 奖励模型分），用于定位 reward hacking |
| `eval_distill20.py` | 蒸馏四方对比的 20 题基准（历史脚本，保留以便交叉验证） |
| `bench200_raw.json` | 12 个权重 × 200 题的原始回复，供复核与追加统计（1.8 MB） |

## 典型用法

```bash
# 全量重跑 200 题基准（12 个权重，约 1.5 h）
python evals/eval_bench200.py

# 只测指定权重（冒号后为 use_moe）
python evals/eval_bench200.py --weights full_sft:0 grpo:0

# 统计与显著性检验（秒级，直接读已有的原始输出）
python evals/analyze_bench200.py --baseline full_sft

# 留出集 PPL 配对检验
python evals/paired_ppl_test.py

# 奖励曲线重建（需要 ../internlm2-1_8b-reward，约 30 min）
python evals/reward_curve_rebuild.py
```

## 已知局限

- 200 题基准的复读率是在**完整回复**上算的。对于把答案留空、只输出思考段的模型
  （如 PPO），这个指标会给出偏低的假象 —— 必须结合退化率与回答长度一起读。
- `reward_curve_rebuild.py` 的 prompt 必须复用训练时的 `RLAIFDataset`（含
  `thinking_ratio`），自行拼 chat 模板会漏掉 `open_thinking`，导致规则奖励里
  `</think>` 相关的两项（合计最高 +1.25）全部拿不到，重建值整体偏低约 1.4。
  脚本里的 GRPO 校准对照就是为了兜住这类错误。
- 逐题复读率的 σ≈23pp，200 题可检出约 6pp 的差异；更小的效应量需要继续扩题量。
