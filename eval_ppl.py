"""在同一批留出数据上直接量各模型的语言建模 loss / 困惑度。

为什么需要它：训练日志里的 "final loss" 是**单个 batch** 的值，噪声极大
（MoE 预训练最后 100 个采样点里，单点在 1.58 ~ 2.41 之间摆，σ=0.165）。
拿两次训练各自的最后一个点比大小是没有意义的。这个脚本让所有模型跑同一
批数据、同样的前向，得到可比的数字。

用法：
    python eval_ppl.py --weights pretrain:0 pretrain:1 --batches 200
    （冒号后是 use_moe）
"""
import os, sys, argparse, math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset.sac_compat  # noqa: F401
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset, SFTDataset

DEV = "cuda:0"


def evaluate(weight, moe, loader, hidden, layers):
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_moe=bool(moe))
    model = MiniMindForCausalLM(cfg)
    path = f"out/{weight}_{hidden}{'_moe' if moe else ''}.pth"
    # strict=False 会静默放过"权重没装进去"这种错（比如把 dense 权重塞进 MoE 骨架，
    # 专家层就全是随机初始化，PPL 会离谱但不报错）。所以把缺失的键数量打出来。
    missing, unexpected = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    real_missing = [k for k in missing if "freqs_c" not in k and "mask" not in k]
    if real_missing:
        print(f"  ⚠ {path} 缺失 {len(real_missing)} 个权重键，例如 {real_missing[:3]}")
    model = model.to(DEV).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())

    tot, ntok = 0.0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for ids, labels in loader:
            ids, labels = ids.to(DEV), labels.to(DEV)
            # 只统计非 padding 位置，逐 token 加权，避免不同 batch 有效长度不同
            res = model(ids, labels=labels)
            mask = (labels[..., 1:] != -100)
            n = int(mask.sum())
            tot += float(res.loss) * n
            ntok += n
    del model
    torch.cuda.empty_cache()
    return tot / max(ntok, 1), n_params


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True, help="形如 pretrain:0 pretrain:1")
    p.add_argument("--data", default="dataset/pretrain_t2t_mini.jsonl")
    p.add_argument("--task", default="pretrain", choices=["pretrain", "sft"],
                   help="pretrain=整段文本都算 loss；sft=对话格式，只对 assistant 回答算 loss")
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=340)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234, help="留出集抽样种子，与训练用的 42 系列不同")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained("model")
    DS = PretrainDataset if args.task == "pretrain" else SFTDataset
    ds = DS(args.data, tok, max_length=args.max_seq_len)
    # 固定随机子集，保证所有模型看到完全相同的数据
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[: args.batches * args.batch_size].tolist()
    sub = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(sub, batch_size=args.batch_size, num_workers=2)
    print(f"留出集：{len(idx)} 条（seed={args.seed}），batch={args.batch_size}\n")

    print(f"{'权重':<22}{'参数量':>11}{'loss':>9}{'PPL':>10}")
    print("-" * 52)
    for spec in args.weights:
        w, moe = spec.split(":")
        loss, n = evaluate(w, int(moe), loader, args.hidden, args.layers)
        tag = w + ("_moe" if int(moe) else "")
        print(f"{tag:<22}{n/1e6:>9.1f}M{loss:>9.4f}{math.exp(loss):>10.2f}", flush=True)
