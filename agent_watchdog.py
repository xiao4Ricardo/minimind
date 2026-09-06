"""Agentic RL 看门狗：崩了就按原参数重新拉起，靠 --from_resume 1 接着跑。

为什么单独写一个而不用 queue_runner：这一阶段是临时追加的单任务，不需要
阶段编排；但 38 小时无人值守必须有崩溃恢复，否则半夜挂掉就白等一天。
完成标记用日志里的最后一步，与其他阶段口径一致。
"""
import os, sys, time, subprocess, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PYEXE = r"C:\Users\Admin\anaconda3\envs\minimind\python.exe"
LOG = os.path.join(ROOT, "agent.log")
WLOG = os.path.join(ROOT, "agent_watchdog.log")
ROWS = 39988
DONE = f"({ROWS}/{ROWS})"
ARGS = ("train_agent.py --from_weight full_sft --save_weight agent --batch_size 1 "
        "--num_generations 2 --max_gen_len 256 --max_seq_len 512 --max_total_len 1300 "
        "--save_interval 400 --num_workers 0 --from_resume 1").split()
MAXRETRY = 40


def log(m):
    line = f"[WD] {datetime.datetime.now():%m-%d %H:%M:%S} {m}"
    print(line, flush=True)
    with open(WLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def running():
    ps = ("if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -like '*train_agent.py*' }) { exit 1 } else { exit 0 }")
    return subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                          capture_output=True).returncode == 1


def done():
    if not os.path.exists(LOG):
        return False
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        return DONE in f.read()


if __name__ == "__main__":
    log(f"看门狗启动，目标 {DONE}")
    tries = 0
    while tries < MAXRETRY:
        if done():
            log("✅ 训练已完成")
            break
        if running():
            time.sleep(60)
            continue
        tries += 1
        log(f"第 {tries} 次拉起 train_agent.py")
        with open(LOG, "a", encoding="utf-8") as lf:
            subprocess.run([PYEXE, "-u"] + ARGS, cwd=os.path.join(ROOT, "trainer"),
                           stdout=lf, stderr=subprocess.STDOUT)
        if done():
            log("✅ 训练完成")
            break
        log("异常退出，60 秒后重试")
        time.sleep(60)
    else:
        log("❌ 超过重试上限，放弃")
