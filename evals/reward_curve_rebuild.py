"""在固定 prompt 集上给各检查点重新打分，重建 CISPO 的奖励曲线。

背景：cispo.log 只保留了第 9291 步之后，起点奖励的记录丢了。但权重都在
（step 0/1610/9290/19502），而奖励是可重算的 —— 用训练时同一套配置和同一个
奖励模型给每个检查点打分即可，不需要重训。

这样重建甚至比原日志更严谨：训练日志里每一步的 prompt 都不同，奖励值里混着
prompt 难度的波动；这里所有检查点用完全相同的 prompt 集。

GRPO 终点作为校准对照：它的日志完整（末 500 步 +0.2539），如果重建值落在附近，
说明这套流程是对的。
"""
import os, sys
# 仓库根目录 = 本文件所在目录的上一级，保证脚本可移植（不写死绝对路径）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import os, sys, re, json, random, math
import dataset.sac_compat  # noqa: F401
import torch, numpy as np
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import LMForRewardModel
from trainer.rollout_engine import create_rollout_engine
from dataset.lm_dataset import RLAIFDataset
from contextlib import nullcontext

DEV = "cuda:0"
N_PROMPT   = 96      # 固定 prompt 数
NUM_GEN    = 6       # 与训练一致
MAX_GEN    = 256     # 与训练一致（日志中回答长度封顶 256）
MAX_SEQ    = 768     # 与训练一致
TEMP       = 0.8     # train_grpo.py 中硬编码
THINK_RATIO= 0.9     # train_grpo.py --thinking_ratio 默认值，必须一致：
                     # 规则奖励里 </think> 那两项合计最高 +1.25，prompt 不开
                     # thinking 就拿不到，重建值会整体低约 1.4 —— 这正是第一版
                     # 校准对照失败的原因

CKPTS = [
    ("step 0（full_sft，RL 起点）", "out/full_sft_768.pth"),
    ("CISPO step 19502（终点）",    "out/backup/cispo_final_768.pth"),
    ("GRPO  step 19502（校准对照）", "out/grpo_768.pth"),
    ("PPO   step 19502",            "out/ppo_actor_768.pth"),
    ("DPO   （对照）",               "out/dpo_lr5e7_768.pth"),
    ("Agentic RL",                  "out/agent_768.pth"),
]

def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return min(cap, (len(grams)-len(set(grams)))*cap*2/len(grams)) if grams else 0.0

def rule_reward(response):
    """train_grpo.py calculate_rewards 里的规则项，逐字复刻"""
    r = 0.0
    r += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    answer = response
    if '</think>' in response:
        thinking, answer_content = response.split('</think>', 1)
        r += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        r += 0.25 if response.count('</think>') == 1 else -0.25
        answer = answer_content.strip()
    r -= rep_penalty(answer)
    return r, answer

tok = AutoTokenizer.from_pretrained("model")

# 固定 prompt 集
# RLAIFDataset 里 create_chat_prompt 会随机决定是否开 thinking、随机加 system
# prompt，所以必须固化成一份列表再给所有检查点用，否则各检查点看到的输入不同。
random.seed(1234)
ds = RLAIFDataset("dataset/rlaif.jsonl", tok, max_length=MAX_SEQ, thinking_ratio=THINK_RATIO)
g = torch.Generator().manual_seed(1234)
sel = torch.randperm(len(ds), generator=g)[:N_PROMPT].tolist()
prompts = [ds[i]["prompt"] for i in sel]
n_think = sum(1 for p in prompts if "<think>" in p)
print(f"其中开启 thinking 的 prompt：{n_think}/{N_PROMPT}")
print(f"固定 prompt 集：{N_PROMPT} 条 × {NUM_GEN} 次采样 = {N_PROMPT*NUM_GEN} 条回答/检查点")
print(f"配置：temperature={TEMP}  max_gen_len={MAX_GEN}  与训练完全一致\n", flush=True)

print("加载奖励模型 internlm2-1_8b-reward …", flush=True)
rm = LMForRewardModel("../internlm2-1_8b-reward", device=DEV, dtype=torch.float16)
print("  完成\n", flush=True)

autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
results = {}
for tag, path in CKPTS:
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8,
                         max_seq_len=MAX_SEQ + MAX_GEN)
    m = MiniMindForCausalLM(cfg)
    miss, _ = m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    rm_keys = [k for k in miss if "freqs_c" not in k and "mask" not in k]
    if rm_keys:
        print(f"  ⚠ {path} 缺失 {len(rm_keys)} 个键")
    m = m.to(DEV).eval().requires_grad_(False)
    eng = create_rollout_engine(engine_type="torch", policy_model=m, tokenizer=tok,
                                device=DEV, autocast_ctx=autocast_ctx)
    torch.manual_seed(1234)          # 采样噪声可比
    per, lens = [], []
    for i, p in enumerate(prompts):
        enc = tok([p], return_tensors="pt", padding=True, return_token_type_ids=False,
                  padding_side="left", add_special_tokens=False).to(DEV)
        enc["input_ids"] = enc["input_ids"][:, -MAX_SEQ:]
        enc["attention_mask"] = enc["attention_mask"][:, -MAX_SEQ:]
        out = eng.rollout(prompt_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                          num_generations=NUM_GEN, max_new_tokens=MAX_GEN, temperature=TEMP)
        matches = re.findall(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", p, re.DOTALL)
        msgs = [{"role": ro, "content": c.strip()} for ro, c in matches]
        for resp in out.completions:
            rr, answer = rule_reward(resp)
            per.append(rr + rm.get_score(msgs, answer))
            lens.append(len(resp.strip()))
        if (i + 1) % 24 == 0:
            print(f"    {tag}  {i+1}/{N_PROMPT}  当前均值 {np.mean(per):+.4f}", flush=True)
    a = np.array(per)
    results[tag] = (a, np.array(lens))
    print(f"  {tag:<30} 奖励均值 {a.mean():+.4f}  回答长度 {np.mean(lens):.1f}  (n={len(a)})\n", flush=True)
    del m, eng; torch.cuda.empty_cache()

print("=" * 78)
print(f"{'检查点':<32}{'奖励均值':>10}{'95% CI':>21}{'回答长度':>10}")
print("-" * 82)
rng = np.random.default_rng(0)
for tag, (a, L) in results.items():
    boot = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{tag:<32}{a.mean():>+10.4f}  [{lo:>+7.4f},{hi:>+7.4f}]{L.mean():>10.1f}")
