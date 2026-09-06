
import os, sys
# 仓库根目录 = 本文件所在目录的上一级，保证脚本可移植（不写死绝对路径）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import sys, io, re, random, torch, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import dataset.sac_compat  # noqa
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import LMForRewardModel
from trainer.rollout_engine import create_rollout_engine
from dataset.lm_dataset import RLAIFDataset

def rep_penalty(text,n=3,cap=0.5):
    t=re.findall(r"\w+|[^\w\s]",text.lower()); g=[tuple(t[i:i+n]) for i in range(len(t)-n+1)]
    return min(cap,(len(g)-len(set(g)))*cap*2/len(g)) if g else 0.0

tok = AutoTokenizer.from_pretrained("model")
random.seed(1234)
ds = RLAIFDataset("dataset/rlaif.jsonl", tok, max_length=768, thinking_ratio=0.9)
g = torch.Generator().manual_seed(1234)
sel = torch.randperm(len(ds), generator=g)[:48].tolist()
prompts=[ds[i]["prompt"] for i in sel]
rm = LMForRewardModel("../internlm2-1_8b-reward", device="cuda:0", dtype=torch.float16)
ac = torch.amp.autocast("cuda", dtype=torch.bfloat16)

print(f"{'模型':<10}{'有</think>':>10}{'答案为空':>10}{'答案长度':>10}{'规则项':>9}{'复读罚':>8}{'RM分':>9}{'总奖励':>9}")
print("-"*76)
for name, path in [("GRPO","out/grpo_768.pth"),("PPO","out/ppo_actor_768.pth"),("Agentic","out/agent_768.pth")]:
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=1024))
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False); m=m.cuda().eval().requires_grad_(False)
    eng = create_rollout_engine(engine_type="torch", policy_model=m, tokenizer=tok, device="cuda:0", autocast_ctx=ac)
    torch.manual_seed(1234)
    has_t=empty=0; alen=[]; rules=[]; reps=[]; rms=[]; tots=[]
    for p in prompts:
        enc=tok([p],return_tensors="pt",padding=True,return_token_type_ids=False,padding_side="left",add_special_tokens=False).to("cuda")
        out=eng.rollout(prompt_ids=enc["input_ids"][:,-768:],attention_mask=enc["attention_mask"][:,-768:],
                        num_generations=6,max_new_tokens=256,temperature=0.8)
        mt=re.findall(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>",p,re.DOTALL)
        msgs=[{"role":r,"content":c.strip()} for r,c in mt]
        for resp in out.completions:
            r = 0.5 if 20<=len(resp.strip())<=800 else -0.5
            ans = resp
            if '</think>' in resp:
                has_t+=1; th,acx = resp.split('</think>',1)
                r += 1.0 if 20<=len(th.strip())<=300 else -0.5
                r += 0.25 if resp.count('</think>')==1 else -0.25
                ans = acx.strip()
            if not ans.strip(): empty+=1
            rp = rep_penalty(ans); s = rm.get_score(msgs, ans)
            alen.append(len(ans)); rules.append(r); reps.append(rp); rms.append(s); tots.append(r-rp+s)
    n=len(tots)
    print(f"{name:<10}{has_t/n:>9.0%}{empty/n:>10.0%}{np.mean(alen):>10.1f}{np.mean(rules):>9.3f}{np.mean(reps):>8.3f}{np.mean(rms):>+9.3f}{np.mean(tots):>+9.3f}")
    del m,eng; torch.cuda.empty_cache()
