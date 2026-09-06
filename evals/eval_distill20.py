"""蒸馏三方对比：dense SFT 基线 / 离线蒸馏 / 在线蒸馏(OPD) / MoE 教师
沿用 eval20.py 的 20 题与 3-gram 复读率口径，贪心解码保证可复现。
额外记录平均长度与零复读题数。"""
import sys, io, re, time, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import dataset.sac_compat  # noqa: F401
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tok = AutoTokenizer.from_pretrained(ROOT + "/model")

# (显示名, 权重文件, 是否 MoE)
MODELS = [("dense_SFT基线", "out/full_sft_768.pth", 0),
          ("离线蒸馏", "out/full_dist_768.pth", 0),
          ("OPD在线蒸馏", "out/opd_768.pth", 0),
          ("MoE教师", "out/full_sft_768_moe.pth", 1)]

QS = ["你好，你是谁？", "请用一句话解释什么是机器学习。", "推荐三种适合新手的编程语言。",
      "水的沸点是多少摄氏度？", "为什么天空是蓝色的？", "写一首关于春天的短诗。",
      "光合作用的原理是什么？", "如何缓解长时间用电脑造成的眼睛疲劳？", "介绍一下长城的历史。",
      "Python 里列表和元组有什么区别？", "怎样煮一碗好吃的番茄鸡蛋面？", "解释一下什么是通货膨胀。",
      "地球到月球有多远？", "给我三个提高睡眠质量的建议。", "什么是区块链？用通俗的话讲。",
      "简述第二次世界大战的起因。", "猫和狗哪个更适合公寓饲养？", "如何用一句话安慰失恋的朋友？",
      "水在什么条件下会结冰？", "写一段自我介绍，用于求职面试。"]


def rep_rate(text, n=3):
    t = re.findall(r"\w+|[^\w\s]", text.lower())
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return (len(g) - len(set(g))) / len(g) if g else 0.0


res, ok, lat = {}, [], {}
for name, path, moe in MODELS:
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=bool(moe))
    m = MiniMindForCausalLM(cfg)
    missing, _ = m.load_state_dict(torch.load(f"{ROOT}/{path}", map_location="cpu"), strict=False)
    rm = [k for k in missing if "freqs_c" not in k and "mask" not in k]
    if rm:
        print(f"  ⚠ {path} 缺失 {len(rm)} 个键，例 {rm[:2]}")
    m = m.cuda().eval()
    outs, t0 = [], time.time()
    for q in QS:
        t = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(t, add_special_tokens=False).input_ids], device="cuda")
        with torch.no_grad():
            o = m.generate(ids, max_new_tokens=256, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        outs.append(tok.decode(o[0][len(ids[0]):], skip_special_tokens=True).strip())
    lat[name] = (time.time() - t0) / len(QS)
    res[name] = outs; ok.append(name)
    print(f"  {name} 完成  ({lat[name]:.2f} s/题)", flush=True)
    del m; torch.cuda.empty_cache()

W = 14
print(f"\n{'问题':<30}" + "".join(f"{n:>{W}}" for n in ok))
print("-" * (30 + W * len(ok)))
tot = {n: 0.0 for n in ok}
for i, q in enumerate(QS):
    row = f"{q[:28]:<30}"
    for n in ok:
        r = rep_rate(res[n][i]); tot[n] += r
        row += f"{r:>{W-1}.1%} "
    print(row)
print("-" * (30 + W * len(ok)))
print(f"{'平均3-gram复读率':<30}" + "".join(f"{tot[n]/len(QS):>{W-1}.1%} " for n in ok))
print(f"{'平均回答字数':<30}" + "".join(f"{sum(len(x) for x in res[n])//len(QS):>{W-1}} " for n in ok))
print(f"{'零复读题数(/20)':<30}" + "".join(f"{sum(1 for i in range(len(QS)) if rep_rate(res[n][i])==0):>{W-1}} " for n in ok))
print(f"{'贪心解码延迟(s/题)':<30}" + "".join(f"{lat[n]:>{W-1}.2f} " for n in ok))

base = ok[0]
print(f"\n=== 逐题对比 {base}（基线）===")
for n in ok[1:]:
    w = l = t_ = 0
    for i in range(len(QS)):
        a, b = rep_rate(res[n][i]), rep_rate(res[base][i])
        if abs(a - b) < 1e-9: t_ += 1
        elif a < b: w += 1
        else: l += 1
    print(f"{n:<14} 优于基线 {w} 题 / 持平 {t_} 题 / 劣于基线 {l} 题")

print("\n=== 样例输出（第 1/5/20 题）===")
for i in [0, 4, 19]:
    print(f"\nQ: {QS[i]}")
    for n in ok:
        print(f"  [{n}] {res[n][i][:160]}")
