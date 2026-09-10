"""MoE 专家路由的直接统计：命中分布、归一化熵、最大/最小比。

为什么需要它：此前"专家没有坍缩"这个结论只有一个间接证据 —— 训练日志里
aux_loss 全程稳定在 0.0040。但 aux_loss 是个标量，它稳定只能说明"负载均衡
损失没有恶化"，不能直接说明每个专家实际分到了多少 token。面试被问到
"你怎么知道没坍缩"时，这是软证据和硬证据的区别。

做法：在每个 MOEFeedForward 的 gate 线性层上挂前向钩子，取出路由 logits，
按模型自身的口径（softmax → top-k）复原专家分配，逐层统计。不修改模型。

指标：
  占比      每个专家分到的 token 比例，num_experts=4 时理想值为 25%
  归一化熵  H / log(num_experts)，1.0 = 完全均匀，0 = 全部挤到一个专家
  max/min   最热专家与最冷专家的占比之比，1.0 最理想

用法：
    python evals/expert_routing.py
    python evals/expert_routing.py --weights pretrain_768_moe.pth
"""
import os, sys, argparse, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import dataset.sac_compat  # noqa: F401
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForward
from dataset.lm_dataset import SFTDataset

DEV = "cuda:0"


def collect(model, batches, num_experts, topk):
    """挂钩 gate，按模型自身口径复原每层的专家分配"""
    counts = {}
    handles = []

    def make_hook(layer_id):
        def hook(_module, _inp, out):
            # out 是 gate 的 logits [N, num_experts]；复刻 forward 里的 softmax→topk
            scores = F.softmax(out.float(), dim=-1)
            _, idx = torch.topk(scores, k=topk, dim=-1, sorted=False)
            c = torch.bincount(idx.flatten(), minlength=num_experts)
            counts[layer_id] = counts.get(layer_id, 0) + c.cpu()
        return hook

    layer_id = 0
    for m in model.modules():
        if isinstance(m, MOEFeedForward):
            handles.append(m.gate.register_forward_hook(make_hook(layer_id)))
            layer_id += 1

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for ids, _ in batches:
            model(ids.to(DEV))

    for h in handles:
        h.remove()
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+",
                    default=["pretrain_768_moe.pth", "full_sft_768_moe.pth"])
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--data", default="dataset/heldout_dpo_chosen.jsonl")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("model")
    ds = SFTDataset(a.data, tok, max_length=340)
    g = torch.Generator().manual_seed(1234)
    idx = torch.randperm(len(ds), generator=g)[: a.batches * a.batch_size].tolist()
    batches = list(DataLoader(Subset(ds, idx), batch_size=a.batch_size, num_workers=0))
    print(f"输入：{len(idx)} 条 × 340 token（留出集，seed=1234）\n")

    for wf in a.weights:
        path = os.path.join(ROOT, "out", wf)
        if not os.path.exists(path):
            print(f"⚠ 跳过 {wf}：不存在")
            continue
        cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=True)
        m = MiniMindForCausalLM(cfg)
        miss, _ = m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        real = [k for k in miss if "freqs_c" not in k and "mask" not in k]
        if real:
            print(f"  ⚠ {wf} 缺失 {len(real)} 个权重键，例 {real[:2]}")
        m = m.to(DEV).eval().requires_grad_(False)
        E, K = cfg.num_experts, cfg.num_experts_per_tok

        counts = collect(m, batches, E, K)
        print(f"=== {wf}   {E} 专家 / top-{K} 路由 ===")
        print(f"{'层':<5}" + "".join(f"{'E'+str(i):>9}" for i in range(E))
              + f"{'归一化熵':>10}{'max/min':>10}")
        print("-" * (5 + 9 * E + 20))
        allc = np.zeros(E)
        for lid in sorted(counts):
            c = counts[lid].numpy().astype(float)
            allc += c
            p = c / c.sum()
            H = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(E)
            print(f"{lid:<5}" + "".join(f"{v:>8.1%} " for v in p)
                  + f"{H:>9.3f}{(p.max()/max(p.min(),1e-9)):>10.2f}")
        p = allc / allc.sum()
        H = -(p[p > 0] * np.log(p[p > 0])).sum() / math.log(E)
        print("-" * (5 + 9 * E + 20))
        print(f"{'全部':<5}" + "".join(f"{v:>8.1%} " for v in p)
              + f"{H:>9.3f}{(p.max()/max(p.min(),1e-9)):>10.2f}")
        dead = int((p < 0.01).sum())
        verdict = ("✓ 未坍缩" if H > 0.95 and dead == 0
                   else ("⚠ 轻度不均" if H > 0.80 else "✗ 坍缩"))
        print(f"  判定：归一化熵 {H:.3f}（1.0=完全均匀）  未使用专家 {dead} 个  {verdict}\n")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
