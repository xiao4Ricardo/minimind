"""配对自助法检验：同一批留出样本上，模型两两之间的 loss 差是否显著。

两个必须做对的地方：
1) SFTDataset 的 pre/post_processing_chat 带随机性（随机加 system prompt、
   随机删空 think 标签），每迭代一次 loader 拿到的张量都不同。所以先把
   batch 固化成一份，四个模型吃完全相同的输入 —— 否则"配对"是假的。
2) 逐条样本保留 loss 和 token 数，才能做配对重采样；配对能消掉样本难度
   带来的方差，比独立比较敏感得多。
"""
import os, sys
# 仓库根目录 = 本文件所在目录的上一级，保证脚本可移植（不写死绝对路径）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import os, sys, math, random
import dataset.sac_compat  # noqa: F401
import torch, numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

DEV = "cuda:0"
MODELS = [("full_sft", 0), ("full_dist", 0), ("opd", 0), ("full_sft_moe", 1),
          ("pretrain", 0), ("pretrain_moe", 1)]

tok = AutoTokenizer.from_pretrained("model")
ds = SFTDataset("dataset/heldout_dpo_chosen.jsonl", tok, max_length=340)
g = torch.Generator().manual_seed(1234)
idx = torch.randperm(len(ds), generator=g)[:1600].tolist()

random.seed(1234)          # 固定数据集内部的随机增强
BATCHES = [(a, b) for a, b in DataLoader(Subset(ds, idx), batch_size=8, num_workers=0)]
ntok_total = int(sum((b[:, 1:] != -100).sum() for _, b in BATCHES))
print(f"固化留出集：{len(idx)} 条 / {len(BATCHES)} 批 / {ntok_total} 个受监督 token\n")

per_seq = {}
for name, moe in MODELS:
    base = name.replace("_moe", "")
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=bool(moe))
    m = MiniMindForCausalLM(cfg)
    miss, _ = m.load_state_dict(
        torch.load(f"out/{base}_768{'_moe' if moe else ''}.pth", map_location="cpu"), strict=False)
    rm = [k for k in miss if "freqs_c" not in k and "mask" not in k]
    if rm:
        print(f"  ⚠ {name} 缺失 {len(rm)} 个权重键，例 {rm[:2]}")
    m = m.to(DEV).eval().requires_grad_(False)
    S, N = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for ids, labels in BATCHES:
            ids, labels = ids.to(DEV), labels.to(DEV)
            logits = m(ids).logits[:, :-1, :].float()
            tgt = labels[:, 1:]
            ls = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                ignore_index=-100, reduction="none").view(tgt.shape)
            msk = (tgt != -100)
            S += (ls * msk).sum(1).tolist()
            N += msk.sum(1).tolist()
    S, N = np.array(S), np.array(N)
    per_seq[name] = (S, N)
    tot = S.sum() / N.sum()
    print(f"{name:<14} loss={tot:.4f}  PPL={math.exp(tot):>6.2f}  有效样本={int((N>0).sum())}", flush=True)
    del m; torch.cuda.empty_cache()

# 所有模型的 token 数必须完全一致，否则说明输入还是不同的
Ns = [tuple(per_seq[n][1]) for n, _ in MODELS]
print(f"\n各模型 token 计数是否完全一致：{len(set(Ns)) == 1}")

print("\n配对自助法（10000 次重采样，按样本重采；负 = 前者更好）")
print(f"{'对比':<36}{'Δloss':>9}{'95% CI':>21}{'p':>9}   结论")
print("-" * 82)
rng = np.random.default_rng(0)
pairs = [("full_dist", "full_sft"), ("opd", "full_sft"), ("opd", "full_dist"),
         ("full_sft_moe", "full_sft"), ("full_dist", "full_sft_moe"),
         ("pretrain_moe", "pretrain"), ("full_sft", "pretrain")]
B = 10000
for a, b in pairs:
    Sa, N = per_seq[a][0], per_seq[a][1]
    Sb = per_seq[b][0]
    keep = N > 0
    Sa, Sb, N = Sa[keep], Sb[keep], N[keep]
    obs = (Sa.sum() - Sb.sum()) / N.sum()
    n = len(N)
    J = rng.integers(0, n, (B, n))
    boot = (Sa[J].sum(1) - Sb[J].sum(1)) / N[J].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    print(f"{a+' vs '+b:<36}{obs:>+9.4f}  [{lo:>+7.4f},{hi:>+7.4f}]{p:>9.4f}   "
          f"{'显著' if lo*hi > 0 else '不显著'}")
