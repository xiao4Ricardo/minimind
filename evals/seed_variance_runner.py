"""种子方差实验：同配置只换随机种子重训，量化训练不确定性。

为什么必须做：报告里所有配对自助法置信区间量的都是**评测**不确定性
（换一批题目考，分数怎么波动），量不到**训练**不确定性（同配置重训一遍，
模型会差多少）。若后者与声称的效应量同量级，那些"显著/不显著"结论就不成立。

先跑离线蒸馏（4.3 h/次，最便宜）拿到训练方差的量级，再决定要不要花
36 小时重跑 CISPO。
"""
import os, sys, time, subprocess, datetime

# 本文件位于 evals/，仓库根是上一级；日志与权重都相对仓库根定位
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYEXE = r"C:\Users\Admin\anaconda3\envs\minimind\python.exe"
WLOG = os.path.join(ROOT, "seed_variance.log")
ROWS, BS = 905718, 16
ITERS = -(-ROWS // BS)          # ceil
DONE = f"Epoch:[1/1]({ITERS}/{ITERS})"

JOBS = [
    ("full_dist_s43", 43),
    ("full_dist_s44", 44),
]
BASE = ("train_distillation.py --teacher_use_moe 1 --student_use_moe 0 "
        "--from_teacher_weight full_sft --from_student_weight full_sft "
        "--epochs 1 --batch_size 16 --num_workers 0 --from_resume 1")


def log(m):
    line = f"[SEED] {datetime.datetime.now():%m-%d %H:%M:%S} {m}"
    print(line, flush=True)
    with open(WLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def running():
    ps = ("if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*train_distillation*' }) { exit 1 } else { exit 0 }")
    return subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                          capture_output=True).returncode == 1


def done(logpath):
    if not os.path.exists(logpath):
        return False
    with open(logpath, encoding="utf-8", errors="replace") as f:
        return DONE in f.read()


if __name__ == "__main__":
    log(f"启动，共 {len(JOBS)} 个任务，完成标记 {DONE}")
    for name, seed in JOBS:
        lp = os.path.join(ROOT, f"seed_{name}.log")
        out = os.path.join(ROOT, "out", f"{name}_768.pth")
        if done(lp) and os.path.exists(out):
            log(f"跳过 {name}：已完成")
            continue
        args = (BASE + f" --save_weight {name} --seed {seed}").split()
        for attempt in range(1, 11):
            if done(lp):
                break
            log(f"{name} (seed={seed}) 第 {attempt} 次拉起")
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
    log("🎉 全部种子任务完成")
