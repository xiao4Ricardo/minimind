"""对 eval_bench200.py 的原始输出做统计与显著性检验。

方法：配对自助法（10000 次重采样）+ 符号检验。配对能消掉题目难度带来的
方差 —— 逐题复读率的标准差在 23 个百分点上下，不配对根本查不出 5pp 级别的差异。

同时报告 200 题与其中旧 20 题子集的结果，用于交叉验证新基准与历史结论一致。
"""
import os, sys, json, re, argparse
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD20 = [
    "你好，你是谁？", "请用一句话解释什么是机器学习。", "推荐三种适合新手的编程语言。",
    "水的沸点是多少摄氏度？", "为什么天空是蓝色的？", "写一首关于春天的短诗。",
    "光合作用的原理是什么？", "如何缓解长时间用电脑造成的眼睛疲劳？", "介绍一下长城的历史。",
    "Python 里列表和元组有什么区别？", "怎样煮一碗好吃的番茄鸡蛋面？", "解释一下什么是通货膨胀。",
    "地球到月球有多远？", "给我三个提高睡眠质量的建议。", "什么是区块链？用通俗的话讲。",
    "简述第二次世界大战的起因。", "猫和狗哪个更适合公寓饲养？", "如何用一句话安慰失恋的朋友？",
    "水在什么条件下会结冰？", "写一段自我介绍，用于求职面试。",
]


def rep_rate(text, n=3):
    t = re.findall(r"\w+|[^\w\s]", text.lower())
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return (len(g) - len(set(g))) / len(g) if g else 0.0


def zh_ratio(text):
    return sum(1 for c in text if '一' <= c <= '鿿') / len(text) if text else 0.0


def paired(a, b, B=10000, seed=0):
    """配对自助法：返回 (Δ均值, CI下界, CI上界, p)"""
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))  # 截断：全零差值会算出 p>1
    return d.mean(), lo, hi, p


def sign_test(a, b):
    d = a - b
    w = int((d < -1e-9).sum())
    l = int((d > 1e-9).sum())
    m = w + l
    if m == 0:
        return w, l, 1.0
    p = 2 * sum(comb(m, k) for k in range(min(w, l) + 1)) / 2 ** m
    return w, l, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="evals/bench200_raw.json")
    ap.add_argument("--baseline", default="full_sft")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.raw), encoding="utf-8") as f:
        D = json.load(f)
    Q = D["questions"]
    R = D["results"]
    maxtok = D["max_new_tokens"]
    names = list(R.keys())
    cats = np.array([q["cat"] for q in Q])
    qtext = [q["q"] for q in Q]
    old_mask = np.array([q in OLD20 for q in qtext])

    # ---- 逐模型指标 ----
    M = {}
    for n in names:
        outs = R[n]
        M[n] = {
            "rep": np.array([rep_rate(o["text"]) for o in outs]),
            "len": np.array([len(o["text"]) for o in outs]),
            "degen": np.array([1.0 if len(o["text"].strip()) < 10 else 0.0 for o in outs]),
            "trunc": np.array([1.0 if o["n_tok"] >= maxtok else 0.0 for o in outs]),
            "zh": np.array([zh_ratio(o["text"]) for o in outs]),
        }

    print(f"200 题基准（{len(Q)} 题 / {len(set(cats))} 类，贪心解码）\n")
    print(f"{'模型':<14}{'复读率':>8}{'长度':>7}{'退化率':>8}{'截断率':>8}{'中文占比':>9}")
    print("-" * 56)
    for n in names:
        m = M[n]
        print(f"{n:<14}{m['rep'].mean():>7.1%}{m['len'].mean():>7.0f}"
              f"{m['degen'].mean():>8.1%}{m['trunc'].mean():>8.1%}{m['zh'].mean():>9.1%}")

    # ---- 对基线的配对检验 ----
    base = args.baseline
    if base not in M:
        print(f"\n⚠ 基线 {base} 不在结果里，跳过检验")
        return
    print(f"\n复读率 vs 基线 {base}（配对自助法 10000 次 + 符号检验）")
    print(f"{'模型':<14}{'Δ(pp)':>8}{'相对':>8}{'95% CI':>19}{'p':>9}{'优/劣':>9}{'符号p':>9}  结论")
    print("-" * 88)
    order = sorted([n for n in names if n != base], key=lambda n: M[n]["rep"].mean())
    for n in order:
        d, lo, hi, p = paired(M[n]["rep"], M[base]["rep"])
        w, l, sp = sign_test(M[n]["rep"], M[base]["rep"])
        rel = d / M[base]["rep"].mean() * 100
        sig = "显著" if lo * hi > 0 else "不显著"
        print(f"{n:<14}{d*100:>+8.2f}{rel:>7.1f}%  [{lo*100:>+6.2f},{hi*100:>+6.2f}]"
              f"{p:>9.4f}{f'{w}/{l}':>9}{sp:>9.4f}  {sig}")

    # ---- 新旧基准交叉验证 ----
    print(f"\n新旧基准交叉验证（同一批输出，200 题 vs 其中的旧 20 题子集）")
    print(f"{'模型':<14}{'200题复读率':>12}{'旧20题':>9}{'200题Δ':>9}{'20题Δ':>8}{'20题CI宽度':>12}{'200题CI宽度':>13}")
    print("-" * 80)
    for n in order:
        d2, lo2, hi2, _ = paired(M[n]["rep"], M[base]["rep"])
        d1, lo1, hi1, _ = paired(M[n]["rep"][old_mask], M[base]["rep"][old_mask])
        print(f"{n:<14}{M[n]['rep'].mean():>11.1%}{M[n]['rep'][old_mask].mean():>9.1%}"
              f"{d2*100:>+9.2f}{d1*100:>+8.2f}{(hi1-lo1)*100:>11.2f}pp{(hi2-lo2)*100:>12.2f}pp")

    # ---- 分类别 ----
    print(f"\n分类别复读率")
    ucats = sorted(set(cats), key=lambda c: list(cats).index(c))
    hdr = f"{'模型':<14}" + "".join(f"{c:>7}" for c in ucats)
    print(hdr); print("-" * len(hdr.encode('gbk', errors='replace')))
    for n in names:
        row = f"{n:<14}"
        for c in ucats:
            row += f"{M[n]['rep'][cats == c].mean():>6.0%} "
        print(row)


if __name__ == "__main__":
    main()
