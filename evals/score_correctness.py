"""在已保存的 200 题原始回复上，程序化打分：指令遵循率 + 事实/计算准确率。

为什么需要它：现有生成质量结论全部压在 3-gram 复读率一个指标上，而这个指标
**奖励少说话** —— 本项目已抓到两例靠"什么都不说"在它上面夺冠的模型（PPO 91%
空答案、pretrain_moe 67% 空答案）。复读率也完全不管答案对不对：

    Q: 水的沸点是多少摄氏度？
    PPO: "...水的沸点确实是100摄氏度。</think>"   ← 答对了，但答案埋在思考段里

这里加的两个指标与"长度"正交，**不会被少说话刷高**：
  指令遵循率  题目里有明确可验证的格式约束（"用一句话"、"三个"、"四行诗"、
              "翻译成英文"），检查输出是否满足
  准确率      事实题与计算题有唯一或近似唯一的正确答案，检查关键词/数值是否出现

打分规则一律写成"命中任一关键词即算对"的宽松形式，宁可高估也不误杀 —— 目的是
横向比较模型，不是给出绝对分数。局限见文件末尾与 README。

用法：
    python evals/score_correctness.py
"""
import os, sys, re, json, argparse
import numpy as np
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _rep3(text, n=3):
    """3-gram 复读率，与 analyze_bench200.py 同口径（在完整回复上算）"""
    t = re.findall(r"\w+|[^\w\s]", text.lower())
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return (len(g) - len(set(g))) / len(g) if g else 0.0


def response_diversity(outs, n=25):
    """不同开头数 / 总题数 —— 检测"答非所问的模式坍缩"。

    为什么需要这第三道防线：官方 PPO 权重在 200 题上复读率 12.2%（全场第 2 好）、
    长度 417（全场最长）、空答案率 0.0%，把"长度守卫"和"空答案守卫"全部绕过，
    但它对 200 道题只产出 44 种不同开头，其中两种占了 105 道 —— 无论问什么都回同
    一篇散文。前两道防线是照着已见过的失败模式设计的，新模式一来就全失效。

    这个指标不依赖答案内容，纯粹看"模型是否在回应不同的输入"。"""
    heads = {" ".join(o["text"].split())[:n] for o in outs}
    return len(heads) / len(outs)


def strip_think(t):
    """只取 </think> 之后的答案部分；没有该标记则整段都算答案。

    必须这样切：PPO 学会了把内容留在思考段里、答案留空，若不切分就会把
    思考段当答案给它送分。"""
    return t.split('</think>', 1)[1].strip() if '</think>' in t else t.strip()


# ---- 准确率规则：(题目, [命中任一即算对的关键词/正则]) ----
# 数值类写成正则以容忍单位与千分位写法
FACT_RULES = {
    "水的沸点是多少摄氏度？": [r"100"],
    "地球到月球有多远？": [r"38\s*万", r"384", r"40\s*万", r"三十八万"],
    "光速是多少？": [r"30\s*万", r"3\s*[×x*]\s*10", r"299", r"三十万"],
    "人体一共有多少块骨头？": [r"206", r"二百零六"],
    "珠穆朗玛峰有多高？": [r"8848", r"8844", r"8,?848", r"8\.8\s*(千米|公里)"],
    "太阳系有几大行星？": [r"八大", r"8\s*大", r"八颗", r"8\s*颗"],
    "中国有多少个省级行政区？": [r"34", r"三十四"],
    "一年有多少天？为什么会有闰年？": [r"365"],
    "血液为什么是红色的？": [r"血红蛋白", r"铁", r"含铁"],
    "世界上最大的海洋是哪个？": [r"太平洋"],
    "长江和黄河哪条更长？": [r"长江"],
    "水在什么条件下会结冰？": [r"0\s*(摄氏度|℃|度)", r"零度", r"零摄氏度"],
    "空气的主要成分是什么？": [r"氮"],
    "为什么会有四季变化？": [r"倾斜", r"公转", r"黄赤交角", r"地轴"],
    "闪电和雷声为什么不同时到达？": [r"光速", r"声速", r"光.{0,6}快", r"传播速度"],
    "彩虹是怎么形成的？": [r"折射", r"色散", r"分解"],
    "为什么天空是蓝色的？": [r"散射", r"瑞利"],
    "月亮为什么有阴晴圆缺？": [r"公转", r"绕地球", r"反射", r"相对位置"],
    "潮汐是怎么产生的？": [r"月球", r"引力", r"万有引力"],
    "为什么铁会生锈？": [r"氧", r"氧化"],
}

MATH_RULES = {
    "如果一件商品原价 200 元，打八折后是多少钱？": [r"160"],
    "小明有 15 个苹果，给了小红 4 个，又买了 7 个，现在有多少个？": [r"18"],
    "一个长方形长 8 米宽 5 米，面积和周长各是多少？": [r"40.{0,20}26|26.{0,20}40"],
    "如果今天是星期三，100 天后是星期几？": [r"星期五", r"周五"],
    "3 个工人 3 小时做 3 个零件，9 个工人 9 小时做多少个？": [r"27"],
    "一瓶水加瓶子共 1.1 元，水比瓶子贵 1 元，瓶子多少钱？": [r"0\.05", r"5\s*分", r"五分"],
    "从 1 加到 100 等于多少？说明你的算法。": [r"5050", r"5,050"],
    "一个数的三倍加 5 等于 26，这个数是多少？": [r"\b7\b", r"等于\s*7", r"是\s*7"],
    "甲比乙大 3 岁，五年后两人年龄和是 33，现在各多少岁？": [r"13.{0,20}10|10.{0,20}13"],
    "抛硬币连续三次都是正面，第四次是正面的概率是多少？": [r"1/2", r"50\s*%", r"0\.5", r"二分之一"],
    "一个班 40 人，25 人喜欢数学，30 人喜欢语文，都喜欢的至少几人？": [r"\b15\b"],
    "时钟在 3 点整，时针和分针的夹角是多少度？": [r"90"],
    "把一根绳子对折三次，会得到几段？": [r"\b8\b", r"八段"],
    "一个正方体有几个面、几条棱、几个顶点？": [r"6.{0,25}12.{0,25}8"],
    "5 台机器 5 分钟造 5 个零件，100 台机器造 100 个要多久？": [r"5\s*分钟", r"五分钟"],
}

# ---- 指令遵循规则：(题目, 判定函数) ----
def _sent_count(t):
    return len([x for x in re.split(r"[。！？!?\n]", t) if x.strip()])

def _list_items(t):
    """数出列举项：有序号/项目符号的行，或顿号分隔"""
    n = len(re.findall(r"(?m)^\s*(?:\d+[.、）)]|[-*•])", t))
    if n == 0:
        n = len(re.findall(r"(?:^|[。\n])\s*[（(]?[一二三四五六七八九1-9][）)、.]", t))
    return n

INSTR_RULES = {
    "请用一句话解释什么是机器学习。":      lambda t: _sent_count(t) <= 2,
    "用一句话总结：机器学习是让计算机从数据中自动发现规律的方法。": lambda t: _sent_count(t) <= 2,
    "用一句话解释什么是团购。":            lambda t: _sent_count(t) <= 2,
    "用一句话说明什么是回收利用。":        lambda t: _sent_count(t) <= 2,
    "如何用一句话安慰失恋的朋友？":        lambda t: _sent_count(t) <= 2,
    "推荐三种适合新手的编程语言。":        lambda t: _list_items(t) >= 3,
    "给我三个提高睡眠质量的建议。":        lambda t: _list_items(t) >= 3,
    "总结一下健康生活的三个要点。":        lambda t: _list_items(t) >= 3,
    "用三个词概括春天。":                  lambda t: len(t) <= 40,
    "写一首关于大海的四行诗。":            lambda t: 3 <= len([x for x in t.split("\n") if x.strip()]) <= 6,
    "请把下面这句话翻译成英文：今天天气很好。": lambda t: sum(c.isascii() and c.isalpha() for c in t) > len(t) * 0.4,
    "给一家新开的咖啡店写一句广告语。":    lambda t: len(t) <= 80,
    "为一个环保公益活动写一句口号。":      lambda t: len(t) <= 80,
    "请把这段话缩短一半：读书能够开阔我们的眼界，增长我们的见识，让我们了解更广阔的世界。": lambda t: len(t) <= 40,
    "把下面内容改成要点形式：早睡早起多喝水多运动少熬夜。": lambda t: _list_items(t) >= 3,
    "用更简洁的话说：由于天气原因导致航班发生了延误的情况。": lambda t: len(t) <= 24,
    "请把这句话扩写：他很努力。": lambda t: len(t) >= 30,
    "怎样管理个人财务？给几条实用建议。": lambda t: _list_items(t) >= 2,
    "写一首关于春天的短诗。": lambda t: len(t) <= 120 and len([x for x in t.splitlines() if x.strip()]) >= 2,
    "写一封感谢老师的短信。": lambda t: len(t) <= 200,
    "写一条推荐一本书的朋友圈文案。": lambda t: len(t) <= 200,
    "把这句话改成疑问句：他明天要去北京。": lambda t: ("？" in t or "?" in t) and len(t) <= 60,
}



# ---- 相关性规则：题目点名两个对象，答案必须同时提到 ----
# 这是第四道守卫，也是唯一能抓住"答非所问式模式坍缩"的行为指标：
# 官方 PPO 无论问什么都回同一篇 AI 伦理散文，既不提猫也不提狗，
# 而它在复读率(12.2%,全场第2)、长度(417,最长)、空答案率(0.0%)三项上全部通过。
RELEVANCE_RULES = {
    "猫和狗哪个更适合公寓饲养？": (["猫"], ["狗"]),
    "读纸质书和电子书各有什么优缺点？": (["纸质", "纸书"], ["电子书", "电子"]),
    "坐高铁和坐飞机出行，该怎么选？": (["高铁", "火车"], ["飞机", "航班"]),
    "自己做饭和点外卖，哪个更划算？": (["做饭", "自己做", "下厨"], ["外卖"]),
    "台式机和笔记本电脑该怎么选？": (["台式"], ["笔记本"]),
    "跑步和游泳哪个减脂效果更好？": (["跑步"], ["游泳"]),
    "租房和买房各有什么考量？": (["租房", "租"], ["买房", "购房"]),
    "线上学习和线下课堂有什么区别？": (["线上", "网课", "在线"], ["线下", "课堂", "面授"]),
    "咖啡和茶，哪个提神效果更好？": (["咖啡"], ["茶"]),
    "手动挡和自动挡汽车怎么选？": (["手动"], ["自动"]),
    "大公司和创业公司，应届生该去哪个？": (["大公司", "大厂"], ["创业", "初创", "小公司"]),
    "早起学习和熬夜学习哪个效率高？": (["早起", "早晨", "清晨"], ["熬夜", "夜晚", "晚上"]),
    "现金支付和移动支付各有什么利弊？": (["现金"], ["移动支付", "电子支付", "手机支付", "扫码"]),
    "中医和西医有什么不同的思路？": (["中医"], ["西医"]),
    "养绿萝和多肉哪个更好打理？": (["绿萝"], ["多肉"]),
    "看电影和看小说，哪种体验更丰富？": (["电影"], ["小说", "书"]),
    "地铁通勤和骑车通勤怎么选？": (["地铁"], ["骑车", "自行车", "单车"]),
    "Windows 和 macOS 的主要区别是什么？": (["windows", "win"], ["mac", "苹果"]),
    "存款和投资理财该怎么平衡？": (["存款", "储蓄"], ["投资", "理财"]),
    "独居和合租各有什么优缺点？": (["独居", "一个人住"], ["合租", "室友"]),
}


def hits_both(ans, pair):
    """答案是否同时提到被比较的两个对象"""
    low = ans.lower()
    return all(any(k.lower() in low for k in side) for side in pair)


def paired(a, b, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    boot = d[rng.integers(0, n, (B, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))  # 截断：全零差值会算出 p>1
    return d.mean(), lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "evals", "bench200_raw.json"))
    ap.add_argument("--baseline", default="full_sft")
    a = ap.parse_args()

    D = json.load(open(a.raw, encoding="utf-8"))
    qs = [q["q"] for q in D["questions"]]
    R = D["results"]

    acc_rules = {**FACT_RULES, **MATH_RULES}
    acc_idx = [i for i, q in enumerate(qs) if q in acc_rules]
    ins_idx = [i for i, q in enumerate(qs) if q in INSTR_RULES]
    rel_idx = [i for i, q in enumerate(qs) if q in RELEVANCE_RULES]
    print(f"准确率题目 {len(acc_idx)} 道（事实 {len(FACT_RULES)} + 计算 {len(MATH_RULES)}）")
    print(f"指令遵循题目 {len(ins_idx)} 道；相关性题目 {len(rel_idx)} 道（须同时提到被比较的两个对象）")
    print("答案一律取 </think> 之后的部分 —— 否则会把思考段当答案给模型送分\n")

    S = {}
    for n, outs in R.items():
        ans = [strip_think(o["text"]) for o in outs]
        acc = np.array([1.0 if any(re.search(p, ans[i]) for p in acc_rules[qs[i]]) else 0.0
                        for i in acc_idx])
        # 空答案必须判负：否则"用一句话"、"写一句口号"这类长度上限约束会被
        # 空串天然满足 —— 实测 PPO 与 pretrain_moe 正是靠这个刷到全场最高的
        # 指令遵循率（60%），而它们 84% / 67% 的答案根本是空的。
        ins = np.array([1.0 if len(ans[i]) >= 10 and INSTR_RULES[qs[i]](ans[i]) else 0.0
                        for i in ins_idx])
        rel = np.array([1.0 if hits_both(ans[i], RELEVANCE_RULES[qs[i]]) else 0.0 for i in rel_idx])
        empty = np.mean([1.0 if len(x) < 10 else 0.0 for x in ans])
        S[n] = {"acc": acc, "ins": ins, "rel": rel, "empty": empty,
                "alen": np.mean([len(x) for x in ans]),
                "div": response_diversity(outs)}

    print(f"{'模型':<14}{'准确率':>9}{'指令遵循':>10}{'相关性':>9}{'答案为空':>10}{'长度':>7}{'多样性':>9}")
    print("-" * 74)
    for n in sorted(S, key=lambda n: -S[n]["acc"].mean()):
        s = S[n]
        warn = "  ⚠坍缩" if s["div"] < 0.5 else ""
        print(f"{n:<14}{s['acc'].mean():>8.1%}{s['ins'].mean():>10.1%}"
              f"{s['rel'].mean():>8.1%}{s['empty']:>10.1%}{s['alen']:>10.0f}{s['div']:>9.1%}{warn}")

    base = a.baseline
    # 退化模型（答案为空占比高）在这两个指标上没有可解释性，单列出来不参与排序比较
    real = [n for n in S if S[n]["empty"] < 0.1]
    degen = [n for n in S if S[n]["empty"] >= 0.1]
    if degen:
        tags = ", ".join("{}({:.0%})".format(n, S[n]["empty"]) for n in degen)
        print(f"\n以下模型答案为空的比例过高，其指标无可解释性，不参与比较：{tags}")

    for key, label in [("acc", "准确率"), ("ins", "指令遵循"), ("rel", "相关性")]:
        print(f"\n{label} vs 基线 {base}（配对自助法，仅非退化模型）")
        print(f"{'模型':<14}{'Δ':>9}{'95% CI':>20}{'p':>9}  结论")
        print("-" * 60)
        for n in sorted(real, key=lambda n: -S[n][key].mean()):
            if n == base:
                continue
            d, lo, hi, p = paired(S[n][key], S[base][key])
            print(f"{n:<14}{d*100:>+8.1f}pp  [{lo*100:>+6.1f},{hi*100:>+6.1f}]{p:>9.4f}  "
                  f"{'显著' if lo*hi > 0 else '不显著'}")

    # 与复读率的对照：证明这两个指标确实与"少说话"正交
    print(f"\n关键对照：复读率最优的三个模型，在准确率上的表现")
    print(f"{'模型':<14}{'复读率排名':>11}{'准确率':>9}{'答案为空':>10}")
    print("-" * 46)
    rep = {n: np.mean([_rep3(o["text"]) for o in R[n]]) for n in S}
    for n in sorted(rep, key=lambda n: rep[n])[:3]:
        rk = sorted(rep, key=lambda x: rep[x]).index(n) + 1
        print(f"{n:<14}{f'第{rk}名 ({rep[n]:.1%})':>13}{S[n]['acc'].mean():>8.1%}{S[n]['empty']:>10.1%}")


if __name__ == "__main__":
    main()
