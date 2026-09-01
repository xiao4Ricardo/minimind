"""训练任务队列：一个阶段跑完自动接下一个，全程无人值守。

替代早先的 chain_moe.bat —— .bat 在多阶段编排上有两个已经踩过的坑：
  1) cmd 按 OEM 代码页解析 .bat，UTF-8 中文注释会破坏后续变量展开；
  2) cmd 的 ">>" 重定向全程持有日志句柄且不授予写共享，别的进程
     往同一个日志追加会静默失败（不报错、不输出）。

每个阶段做四件事：等前置条件 -> （可选）显存探针定 batch_size ->
带重试地跑训练 -> 用日志里的完成标记确认真的跑完了。

启动：powershell -c "Start-Process python -ArgumentList 'queue_runner.py' -WindowStyle Hidden"
"""
import os, sys, time, math, subprocess, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PYEXE = r"C:\Users\Admin\anaconda3\envs\minimind\python.exe"
QLOG = os.path.join(ROOT, "queue.log")
MAXRETRY = 30
SFT_ROWS, RLAIF_ROWS = 905718, 19502

STAGES = [
    {
        "name": "MoE full_sft",
        "script": "train_full_sft.py",
        "log": "sft_moe.log",
        "rows": SFT_ROWS, "epochs": 2, "batch_size": 6,
        "args": "--use_moe 1 --from_weight pretrain --save_weight full_sft "
                "--batch_size {bs} --accumulation_steps 3 --num_workers 4 --from_resume 1",
        "requires": ["out/pretrain_768_moe.pth"],
        "wait_exit": ["train_pretrain.py"],
        "produces": "out/full_sft_768_moe.pth",
    },
    {
        "name": "离线蒸馏 MoE->dense",
        "script": "train_distillation.py",
        "log": "distill_moe.log",
        "rows": SFT_ROWS, "epochs": 1,
        "probe": "--stage distill --candidates 16,12,8,6,4 --seq 340 "
                 "--student_weight full_sft --teacher_weight full_sft",
        "args": "--teacher_use_moe 1 --student_use_moe 0 --from_teacher_weight full_sft "
                "--from_student_weight full_sft --save_weight full_dist --epochs 1 "
                "--batch_size {bs} --num_workers 4 --from_resume 1",
        "requires": ["out/full_sft_768_moe.pth", "out/full_sft_768.pth"],
        "wait_exit": [],
        "produces": "out/full_dist_768.pth",
    },
    {
        "name": "OPD 在线蒸馏",
        "script": "train_opd.py",
        "log": "opd.log",
        "rows": RLAIF_ROWS, "epochs": 1, "batch_size": 1,
        "args": "--teacher_use_moe 1 --from_teacher_weight full_sft --from_weight full_sft "
                "--save_weight opd --batch_size {bs} --num_generations 6 --max_seq_len 512 "
                "--max_gen_len 256 --num_workers 2 --from_resume 1",
        "requires": ["out/full_sft_768_moe.pth", "out/full_sft_768.pth"],
        "wait_exit": [],
        "produces": "out/opd_768.pth",
    },
]


def log(msg):
    line = f"[QUEUE] {datetime.datetime.now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(QLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def running(script):
    """某个训练脚本是否还在跑"""
    ps = ("if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          f"Where-Object {{ $_.CommandLine -like '*{script}*' }}) {{ exit 1 }} else {{ exit 0 }}")
    return subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                          capture_output=True).returncode == 1


def marker_in(logpath, marker):
    if not os.path.exists(logpath):
        return False
    with open(logpath, "r", encoding="utf-8", errors="replace") as f:
        return marker in f.read()


def probe_bs(stage):
    log(f"  显存探针启动：{stage['probe']}")
    r = subprocess.run([PYEXE, "-u", "probe_mem.py"] + stage["probe"].split(),
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    for ln in (r.stdout or "").splitlines():
        if ln.strip():
            log("  " + ln.strip())
    for ln in reversed((r.stdout or "").splitlines()):
        if ln.startswith("CHOSEN_BS="):
            return int(ln.split("=")[1])
    # 探针本身崩了也不能拖垮队列：把 stderr 记下来便于排查，然后退到保守值。
    # bs=4 时学生 64M 训练态 + 教师 198M 推理态约 2.6 GB，余量充足。
    for ln in (r.stderr or "").splitlines()[-15:]:
        if ln.strip():
            log("  [probe stderr] " + ln.strip()[:200])
    log("  探针未给出结果，回退 bs=4")
    return 4


def run_stage(stage):
    name = stage["name"]
    logpath = os.path.join(ROOT, stage["log"])

    # ---- 已经跑完过就跳过（队列可以安全地重启） ----
    bs_known = stage.get("batch_size")
    if bs_known:
        iters = math.ceil(stage["rows"] / bs_known)
        done_marker = f"Epoch:[{stage['epochs']}/{stage['epochs']}]({iters}/{iters})"
        if marker_in(logpath, done_marker) and os.path.exists(os.path.join(ROOT, stage["produces"])):
            log(f"跳过「{name}」：日志已有完成标记且权重已存在")
            return True

    # ---- 等前置进程退出 ----
    for script in stage["wait_exit"]:
        if running(script):
            log(f"「{name}」等待 {script} 退出…")
            while running(script):
                time.sleep(30)
            log(f"  {script} 已退出")
            time.sleep(30)  # 让它把权重落盘写完

    # ---- 等前置权重就绪 ----
    for req in stage["requires"]:
        p = os.path.join(ROOT, req)
        if not os.path.exists(p):
            log(f"「{name}」等待前置权重 {req} …")
            while not os.path.exists(p):
                time.sleep(60)
        log(f"  前置就绪 {req} ({os.path.getsize(p)/1e6:.0f} MB)")

    # ---- 定 batch_size ----
    bs = stage.get("batch_size") or probe_bs(stage)
    iters = math.ceil(stage["rows"] / bs)
    done_marker = f"Epoch:[{stage['epochs']}/{stage['epochs']}]({iters}/{iters})"
    log(f"启动「{name}」 bs={bs}  共 {iters}×{stage['epochs']} 步  完成标记 {done_marker}")

    cmd = [PYEXE, "-u", stage["script"]] + stage["args"].format(bs=bs).split()
    for attempt in range(1, MAXRETRY + 1):
        log(f"  第 {attempt} 次拉起 {stage['script']}")
        with open(logpath, "a", encoding="utf-8") as lf:
            subprocess.run(cmd, cwd=os.path.join(ROOT, "trainer"),
                           stdout=lf, stderr=subprocess.STDOUT)
        if marker_in(logpath, done_marker):
            log(f"✅「{name}」完成（共拉起 {attempt} 次）")
            return True
        log(f"  异常退出，30 秒后重试")
        time.sleep(30)
    log(f"❌「{name}」连续 {MAXRETRY} 次失败，队列中止")
    return False


if __name__ == "__main__":
    log("=" * 60)
    log(f"队列启动，共 {len(STAGES)} 个阶段：" + " -> ".join(s["name"] for s in STAGES))
    for i, stage in enumerate(STAGES, 1):
        log(f"--- 阶段 {i}/{len(STAGES)}：{stage['name']} ---")
        if not run_stage(stage):
            log("队列因失败中止，后续阶段不再执行")
            sys.exit(1)
    log("🎉 全部阶段完成")
