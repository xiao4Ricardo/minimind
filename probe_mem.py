"""训练前的显存探针：在真实模型上跑几步，量出峰值显存，挑一个不会触发
NVIDIA 系统内存回退的 batch_size。

为什么需要它：这张 8GB 卡显存不够时不会抛 OOM，而是静默退回系统内存，
步耗时恶化十几到二十倍（实测 MoE 预训练 bs=32 时 4.88 秒/步 vs bs=16 的
0.24 秒/步）。训练照跑，只是慢二十倍，不测就发现不了。

用法：
    python probe_mem.py --stage distill --candidates 16,12,8,6,4
输出最后一行固定为 CHOSEN_BS=<n>，供 queue_runner.py 读取。
"""
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset.sac_compat  # noqa: F401
import torch
import torch.nn.functional as F
from torch import optim
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

DEV = "cuda:0"


def build(hidden, layers, moe, weight, train):
    cfg = MiniMindConfig(hidden_size=hidden, num_hidden_layers=layers, use_moe=moe)
    m = MiniMindForCausalLM(cfg)
    if weight:
        path = f"out/{weight}_{hidden}{'_moe' if moe else ''}.pth"
        m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    m = m.to(DEV)
    if train:
        m.train()
    else:
        m.eval(); m.requires_grad_(False)
    return m, cfg


def step_distill(student, teacher, opt, scaler, bs, seq):
    ids = torch.randint(0, 6400, (bs, seq), device=DEV)
    labels = ids.clone()
    loss_mask = (labels[..., 1:] != -100).float()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        s_logits = student(ids).logits[..., :-1, :].contiguous()
    with torch.no_grad():
        t_logits = teacher(ids).logits[..., :-1, :].contiguous()[..., :s_logits.size(-1)]
    flat = loss_mask.view(-1)
    ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                         labels[..., 1:].contiguous().view(-1),
                         ignore_index=-100, reduction="none")
    ce = (ce * flat).sum() / (flat.sum() + 1e-8)
    sm = s_logits.view(-1, s_logits.size(-1))[flat == 1]
    tm = t_logits.view(-1, t_logits.size(-1))[flat == 1]
    with torch.no_grad():
        tp = F.softmax(tm / 1.5, dim=-1)
    kl = F.kl_div(F.log_softmax(sm / 1.5, dim=-1), tp, reduction="batchmean") * (1.5 ** 2)
    loss = 0.5 * ce + 0.5 * kl
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)


def step_plain(model, opt, scaler, bs, seq, moe):
    ids = torch.randint(0, 6400, (bs, seq), device=DEV)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        res = model(ids, labels=ids)
        loss = res.loss + (res.aux_loss if moe else 0)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)


def measure(stage, bs, seq, args):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if stage == "distill":
        student, _ = build(args.student_hidden, args.student_layers, False, args.student_weight, True)
        teacher, _ = build(args.teacher_hidden, args.teacher_layers, True, args.teacher_weight, False)
        opt = optim.AdamW(student.parameters(), lr=5e-6)
        scaler = torch.amp.GradScaler(enabled=False)
        for _ in range(5):
            step_distill(student, teacher, opt, scaler, bs, seq)
        peak = torch.cuda.max_memory_reserved()
        del student, teacher, opt
    else:
        moe = bool(args.student_moe)
        model, _ = build(args.student_hidden, args.student_layers, moe, args.student_weight, True)
        opt = optim.AdamW(model.parameters(), lr=1e-5)
        scaler = torch.amp.GradScaler(enabled=False)
        for _ in range(5):
            step_plain(model, opt, scaler, bs, seq, moe)
        peak = torch.cuda.max_memory_reserved()
        del model, opt
    torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["distill", "plain"])
    p.add_argument("--candidates", required=True, help="逗号分隔，从大到小")
    p.add_argument("--seq", type=int, default=340)
    p.add_argument("--headroom_gb", type=float, default=0.45, help="留给桌面波动的余量")
    p.add_argument("--student_hidden", type=int, default=768)
    p.add_argument("--student_layers", type=int, default=8)
    p.add_argument("--student_moe", type=int, default=0)
    p.add_argument("--student_weight", default="full_sft")
    p.add_argument("--teacher_hidden", type=int, default=768)
    p.add_argument("--teacher_layers", type=int, default=8)
    p.add_argument("--teacher_weight", default="full_sft")
    args = p.parse_args()

    free, total = torch.cuda.mem_get_info()
    budget = free - args.headroom_gb * 1e9
    print(f"[probe] 空闲 {free/1e9:.2f} GB / 总 {total/1e9:.2f} GB，"
          f"预算上限 {budget/1e9:.2f} GB（留 {args.headroom_gb} GB 余量）", flush=True)

    chosen = None
    for bs in [int(x) for x in args.candidates.split(",")]:
        try:
            peak = measure(args.stage, bs, args.seq, args)
        except RuntimeError as e:
            print(f"[probe] bs={bs:<3} 失败: {str(e)[:70]}", flush=True)
            torch.cuda.empty_cache()
            continue
        ok = peak <= budget
        print(f"[probe] bs={bs:<3} seq={args.seq} 峰值保留 {peak/1e9:.2f} GB  "
              f"{'✅ 通过' if ok else '❌ 超预算'}", flush=True)
        if ok:
            chosen = bs
            break
    if chosen is None:
        chosen = int(args.candidates.split(",")[-1])
        print(f"[probe] 全部超预算，回退到最小候选 {chosen}", flush=True)
    print(f"CHOSEN_BS={chosen}")
