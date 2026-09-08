"""RL 训练的种子方差实验：3 个全新种子，配置逐字一致。

为什么不复用原始那次（seed=42）作为三点之一：RLAIFDataset 用全局 random 决定
每条样本是否开 thinking，而 num_workers>0 时每个 worker 有独立 RNG —— num_workers
因此是第二个随机源。原始 CISPO 那次用的 num_workers 已无从查证，拿它当对照会把
"种子差异"和"worker 数差异"混在一起。三个全新种子、num_workers 统一为 0，
方差就是干净的。

配置对齐原始那次（从 cispo.log 反推）：
  batch_size=1（19502 步 = rlaif 19502 行）
  max_gen_len=256（日志中回答长度封顶 256）
  num_generations=6, loss_type=cispo, lr=3e-7（均为默认值）
"""
import os, sys, time, subprocess, datetime

# 本文件位于 evals/，仓库根是上一级；日志与权重都相对仓库根定位
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYEXE = r"C:\Users\Admin\anaconda3\envs\minimind\python.exe"
WLOG = os.path.join(ROOT, "seed_variance_rl.log")
DONE = "Epoch:[1/1](19502/19502)"
SEEDS = [43, 44, 45]
BASE = ("train_grpo.py --loss_type cispo --from_weight full_sft "
        "--batch_size 1 --max_gen_len 256 --num_generations 6 "
        "--num_workers 0 --save_interval 500 --from_resume 1")


def log(m):
    line = f"[RLSEED] {datetime.datetime.now():%m-%d %H:%M:%S} {m}"
    print(line, flush=True)
    with open(WLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done(p):
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8", errors="replace") as f:
        return DONE in f.read()


if __name__ == "__main__":
    log(f"启动：CISPO × {len(SEEDS)} 个种子 {SEEDS}，每个约 18 h，合计约 {18*len(SEEDS)} h")
    for seed in SEEDS:
        name = f"cispo_s{seed}"
        lp = os.path.join(ROOT, f"seed_{name}.log")
        out = os.path.join(ROOT, "out", f"{name}_768.pth")
        if done(lp) and os.path.exists(out):
            log(f"跳过 {name}：已完成")
            continue
        args = (BASE + f" --save_weight {name} --seed {seed}").split()
        for attempt in range(1, 16):
            if done(lp):
                break
            log(f"{name} 第 {attempt} 次拉起")
            with open(lp, "a", encoding="utf-8") as lf:
                subprocess.run([PYEXE, "-u"] + args, cwd=os.path.join(ROOT, "trainer"),
                               stdout=lf, stderr=subprocess.STDOUT)
            if done(lp):
                log(f"✅ {name} 完成")
                break
            log(f"{name} 异常退出，60 秒后重试")
            time.sleep(60)
        else:
            log(f"❌ {name} 重试耗尽，中止")
            sys.exit(1)
    log("🎉 三个种子全部完成")
