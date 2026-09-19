"""字幕断句核心:全局动态规划选择分段,使加权总代价最低。

代价项(均可加权):
  - 镜头切点偏差: 分段边界偏离最近镜头切点的距离(归一化), 以及分段跨越切点的惩罚
  - 行长不均:     两行长度差(归一化)
  - 不自然停顿:   断点处词间间隔过小(归一化)

硬约束:
  - 每段最多 2 行, 每行不超过 max_chars_per_line 个字符
  - 阅读速度(字符/秒)不超过 max_cps
  - 每段时长在 [min_duration, max_duration] 内
  - 不可拆短语(phrase)作为原子单元, 不得被分段或换行拆开
  - 边界约束(boundaries): 指定词间隙必须断开(must_break)或禁止断开(no_break),
    在动态规划中作为硬条件生效; 不可拆短语内部的词间隙不允许设置
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    max_chars_per_line: int = 42
    max_cps: float = 20.0
    min_duration: float = 0.8
    max_duration: float = 7.0
    pause_threshold: float = 0.3      # 词间间隔小于该值视为不自然断点
    shot_dev_cap: float = 0.5         # 镜头切点偏差归一化上限(秒)
    crossing_penalty: float = 1.0     # 分段跨越一个镜头切点的惩罚(归一化)
    w_shot: float = 1.0
    w_imbalance: float = 1.0
    w_pause: float = 1.0


@dataclass
class Unit:
    """原子单元: 单个词或一个不可拆短语。"""
    word_indices: list[int]
    text: str
    start: float
    end: float


def build_units(words: list[dict], phrases: list[list[int]]) -> list[Unit]:
    """把词序列与不可拆短语(词下标区间)合并为原子单元序列。"""
    n = len(words)
    owner: dict[int, list[int]] = {}
    for ph in phrases:
        if not ph:
            continue
        if ph != list(range(ph[0], ph[-1] + 1)):
            raise ValueError(f"短语下标不连续: {ph}")
        for i in ph:
            if i < 0 or i >= n:
                raise ValueError(f"短语下标越界: {i}")
            if i in owner:
                raise ValueError(f"短语重叠于词 {i}")
        for i in ph:
            owner[i] = ph

    units: list[Unit] = []
    i = 0
    while i < n:
        if i in owner:
            ph = owner[i]
            units.append(Unit(
                word_indices=ph,
                text=" ".join(words[k]["text"] for k in ph),
                start=words[ph[0]]["start"],
                end=words[ph[-1]]["end"],
            ))
            i = ph[-1] + 1
        else:
            units.append(Unit(
                word_indices=[i],
                text=words[i]["text"],
                start=words[i]["start"],
                end=words[i]["end"],
            ))
            i += 1
    return units


def validate_boundaries(boundaries: list[dict], n_words: int,
                        phrases: list[list[int]], units: list[Unit]):
    """校验词间隙边界约束, 返回 (must_break, no_break) 两个单元边界集合。

    边界下标 k 表示词 k-1 与词 k 之间的间隙(1 <= k <= n_words-1);
    单元边界 u 位于单元 u-1 与单元 u 之间。不可拆短语内部的间隙不得设置。
    """
    mb: set[int] = set()
    nb: set[int] = set()
    if not boundaries:
        return mb, nb
    inner: set[int] = set()  # 短语内部的词间隙
    for ph in phrases:
        for k in range(ph[0] + 1, ph[-1] + 1):
            inner.add(k)
    seen: dict[int, str] = {}
    for b in boundaries:
        k, t = b.get("index"), b.get("type")
        if t not in ("must_break", "no_break"):
            raise ValueError(f"未知边界约束类型: {t}")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > n_words - 1:
            raise ValueError(f"边界下标越界: {k}(合法范围 1–{n_words - 1})")
        if k in inner:
            raise ValueError(f"边界 {k} 位于不可拆短语内部, 不能设置边界约束")
        if k in seen and seen[k] != t:
            raise ValueError(f"边界 {k} 同时被设置为必须断开与禁止断开")
        seen[k] = t
    # 词 k 不在短语内部, 必为某单元首词
    start_to_unit = {u.word_indices[0]: i for i, u in enumerate(units)}
    for k, t in seen.items():
        (mb if t == "must_break" else nb).add(start_to_unit[k])
    return mb, nb


def layout_segment(units: list[Unit], a: int, b: int, max_chars: int):
    """尝试把 units[a:b] 排成至多两行, 返回 (lines, imbalance) 或 None。

    单元不可拆行; 枚举所有换行位置, 取行长差最小者。
    """
    texts = [u.text for u in units[a:b]]
    if any(len(t) > max_chars for t in texts):
        return None

    def join(ts):
        return " ".join(ts)

    best = None
    # 单行
    one = join(texts)
    if len(one) <= max_chars:
        best = ([one], 0.0)
    # 两行
    for k in range(1, len(texts)):
        l1, l2 = join(texts[:k]), join(texts[k:])
        if len(l1) <= max_chars and len(l2) <= max_chars:
            imb = abs(len(l1) - len(l2)) / max_chars
            if best is None or imb < best[1]:
                best = ([l1, l2], imb)
    return best


def segment_violations(units: list[Unit], a: int, b: int, cfg: Config) -> list[str]:
    """列出段 units[a..b](闭区间)违反的硬约束(空列表表示可行)。"""
    seg = units[a:b + 1]
    dur = seg[-1].end - seg[0].start
    chars = sum(len(u.text) for u in seg) + len(seg) - 1
    v = []
    if dur > cfg.max_duration + 1e-9:
        v.append(f"时长 {dur:.2f}s 超过上限 {cfg.max_duration}s")
    if dur < cfg.min_duration - 1e-9:
        v.append(f"时长 {dur:.2f}s 低于下限 {cfg.min_duration}s")
    if dur > 1e-9 and chars / dur > cfg.max_cps + 1e-9:
        v.append(f"阅读速度 {chars / dur:.1f} 字/秒 超过上限 {cfg.max_cps}")
    if layout_segment(units, a, b + 1, cfg.max_chars_per_line) is None:
        v.append(f"无法在 2 行 x {cfg.max_chars_per_line} 字符内排下")
    return v


def boundary_shot_cost(t: float, shots: list[float], cfg: Config) -> float:
    if not shots:
        return 0.0
    d = min(abs(t - s) for s in shots)
    return min(d, cfg.shot_dev_cap) / cfg.shot_dev_cap


def crossing_count(start: float, end: float, shots: list[float]) -> int:
    return sum(1 for s in shots if start + 1e-9 < s < end - 1e-9)


def pause_cost(gap: float, cfg: Config) -> float:
    if gap >= cfg.pause_threshold:
        return 0.0
    return min(1.0, (cfg.pause_threshold - gap) / cfg.pause_threshold)


def _run_dp(units: list[Unit], feasible: list[list[bool]], cost_parts: list[list[dict]],
            cfg: Config, shots: list[float], mb: set[int], nb: set[int]):
    """全局 DP: dp[j] = 前 j 个单元的最小加权总代价。

    边界约束(单元边界 u 位于单元 u-1 与 u 之间)作为硬条件:
      - u ∈ mb: 必须断开 —— 任何分段不得跨越该边界(转移 i→j 不允许 i < u < j)
      - u ∈ nb: 禁止断开 —— 单元 u-1 与 u 必须同段(不允许以 u 为段首)
    """
    INF = float("inf")
    n = len(units)
    # lo[j]: 转移 i→j 合法要求 i >= lo[j](不得跨过小于 j 的 must_break 边界)
    lo = [0] * (n + 1)
    for j in range(1, n + 1):
        lo[j] = max(lo[j - 1], j - 1 if (j - 1) in mb else 0)
    dp = [INF] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0.0
    for j in range(1, n + 1):
        for i in range(lo[j], j):
            if i and i in nb:
                continue
            if dp[i] == INF or not feasible[i][j - 1]:
                continue
            p = cost_parts[i][j - 1]
            c_shot = p["crossing"] * cfg.crossing_penalty
            c_pause = 0.0
            if i > 0:  # 段首边界(非全文开头): 切点偏差 + 停顿
                t = units[i].start
                c_shot += boundary_shot_cost(t, shots, cfg)
                c_pause = pause_cost(units[i].start - units[i - 1].end, cfg)
            total = (cfg.w_shot * c_shot
                     + cfg.w_imbalance * p["imbalance"]
                     + cfg.w_pause * c_pause)
            if dp[i] + total < dp[j]:
                dp[j] = dp[i] + total
                prev[j] = i
    return dp, prev


def solve(words: list[dict], phrases: list[list[int]], shots: list[float],
          cfg: Config, boundaries: list[dict] | None = None) -> dict:
    if not words:
        return {"ok": True, "total_cost": 0.0, "segments": []}
    for i, w in enumerate(words):
        if w["end"] < w["start"]:
            raise ValueError(f"词 {i} 的结束时间早于开始时间")
        if i and w["start"] < words[i - 1]["start"] - 1e-9:
            raise ValueError(f"词 {i} 的开始时间早于前一个词")

    units = build_units(words, phrases)
    n = len(units)
    shots = sorted(shots)
    mb, nb = validate_boundaries(boundaries or [], len(words), phrases, units)

    # seg_cost[a][b]: 段 units[a:b+1] 的可行性与代价分解
    INF = float("inf")
    feasible = [[False] * n for _ in range(n)]
    cost_parts = [[None] * n for _ in range(n)]
    for a in range(n):
        for b in range(a, n):
            if units[b].end - units[a].start > cfg.max_duration + 1e-9:
                break  # 再延长只会更超时长
            if segment_violations(units, a, b, cfg):
                continue
            lay = layout_segment(units, a, b, cfg.max_chars_per_line)
            lines, imb = lay
            start, end = units[a].start, units[b].end
            dur = end - start
            chars = sum(len(u.text) for u in units[a:b + 1]) + (b - a)
            cross = crossing_count(start, end, shots)
            feasible[a][b] = True
            cost_parts[a][b] = {
                "lines": lines,
                "imbalance": imb,
                "crossing": cross,
                "duration": dur,
                "cps": chars / dur if dur > 1e-9 else 0.0,
            }

    dp, prev = _run_dp(units, feasible, cost_parts, cfg, shots, mb, nb)

    if dp[n] == INF:
        return _conflict(units, feasible, cost_parts, cfg, shots, mb, nb)

    # 回溯分段
    bounds = []
    j = n
    while j > 0:
        i = prev[j]
        bounds.append((i, j - 1))
        j = i
    bounds.reverse()

    segments = []
    for a, b in bounds:
        p = cost_parts[a][b]
        start, end = units[a].start, units[b].end
        c_shot = p["crossing"] * cfg.crossing_penalty
        c_pause = 0.0
        if a > 0:
            c_shot += boundary_shot_cost(start, shots, cfg)
            c_pause = pause_cost(units[a].start - units[a - 1].end, cfg)
        c_shot_w = cfg.w_shot * c_shot
        c_imb_w = cfg.w_imbalance * p["imbalance"]
        c_pause_w = cfg.w_pause * c_pause
        segments.append({
            "word_range": [units[a].word_indices[0], units[b].word_indices[-1]],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(p["duration"], 3),
            "cps": round(p["cps"], 1),
            "lines": p["lines"],
            "cost": {
                "shot": round(c_shot_w, 4),
                "imbalance": round(c_imb_w, 4),
                "pause": round(c_pause_w, 4),
                "total": round(c_shot_w + c_imb_w + c_pause_w, 4),
            },
        })
    return {"ok": True, "total_cost": round(dp[n], 4), "segments": segments}


def _conflict(units: list[Unit], feasible: list[list[bool]], cost_parts: list[list[dict]],
              cfg: Config, shots: list[float], mb: set[int], nb: set[int]) -> dict:
    """无可行方案时定位冲突: 若由边界约束所致, 返回首个不可行边界及具体原因。"""
    n = len(units)
    INF = float("inf")
    dp0, _ = _run_dp(units, feasible, cost_parts, cfg, shots, set(), set())
    if dp0[n] == INF:
        # 去掉全部边界约束仍不可行: 与约束无关, 走通用冲突定位
        return _plain_conflict(units, dp0, cfg)

    # 边界约束导致不可行: 按单元边界升序逐个加入, 找首个使问题不可行的约束
    ordered = sorted([(u, "must_break") for u in mb] + [(u, "no_break") for u in nb])
    cur_mb: set[int] = set()
    cur_nb: set[int] = set()
    culprit = None
    for u, t in ordered:
        (cur_mb if t == "must_break" else cur_nb).add(u)
        dp, _ = _run_dp(units, feasible, cost_parts, cfg, shots, cur_mb, cur_nb)
        if dp[n] == INF:
            culprit = (u, t)
            break
    if culprit is None:  # 完整集合已知不可行, 理论上不会到达
        return _plain_conflict(units, dp0, cfg)

    u, t = culprit
    k = units[u].word_indices[0]  # 词间隙下标
    if t == "must_break":
        s, e, a, reasons = _must_break_range(units, cfg, cur_mb, cur_nb)
        side = "右侧" if s == u else "左侧"
        ws, we = units[s].word_indices[0], units[e - 1].word_indices[-1]
        head = f"词 {k - 1} 与词 {k} 之间必须断开后, {side}区间(词 {ws}–{we})无可行分段"
    else:
        a, reasons = _no_break_reason(units, cfg, u, cur_mb, cur_nb)
        head = f"词 {k - 1} 与词 {k} 之间禁止断开, 两侧内容必须落在同一段"
    wi = units[a].word_indices
    return {
        "ok": False,
        "conflict": {
            "word_range": [wi[0], wi[-1]],
            "start": round(units[a].start, 3),
            "end": round(units[a].end, 3),
            "text": units[a].text,
            "reasons": [head] + reasons,
            "boundary": {"index": k, "type": t},
        },
    }


def _range_conflict(units: list[Unit], cfg: Config, s: int, e: int, nb: set[int]):
    """区间 [s, e) 必须独立分段(内部遵守禁止断开 nb)时,
    返回 (首个不可覆盖单元下标, 违例列表); 区间可行则返回 None。"""
    INF = float("inf")
    m = e - s
    dp = [INF] * (m + 1)
    dp[0] = 0.0
    for j in range(1, m + 1):
        for i in range(j):
            if i and (s + i) in nb:
                continue
            if dp[i] == INF or segment_violations(units, s + i, s + j - 1, cfg):
                continue
            dp[j] = 0.0
            break
    if dp[m] != INF:
        return None
    # 首个不可覆盖单元 = 最后一个可达切点之后的单元
    # (禁止断开可能使中间切点暂时不可达、随后又恢复, 不能取首个不可达位置)
    r = max(j for j in range(m + 1) if dp[j] != INF)
    a = s + r
    reasons = segment_violations(units, a, a, cfg)
    if not reasons:
        # 单元单独可行, 但与前方可达边界拼接均失败: 汇总最近一段的违例
        for i in range(a - 1, s - 1, -1):
            if dp[i - s] != INF:
                reasons = segment_violations(units, i, a, cfg)
                break
        if not reasons:
            reasons = ["与相邻单元组合后无可行布局"]
    return a, reasons


def _must_break_range(units: list[Unit], cfg: Config, mb: set[int], nb: set[int]):
    """must_break 边界把序列切成必须各自独立分段的区间, 返回首个不可行区间
    (s, e, 冲突单元下标, 违例列表)。"""
    n = len(units)
    cuts = [0] + sorted(mb) + [n]
    for s, e in zip(cuts, cuts[1:]):
        r = _range_conflict(units, cfg, s, e, nb)
        if r:
            a, reasons = r
            return s, e, a, reasons
    return 0, n, 0, ["无可行分段"]  # 完整集合已知不可行, 理论上不会到达


def _no_break_reason(units: list[Unit], cfg: Config, u: int, mb: set[int], nb: set[int]):
    """禁止断开边界 u: 沿连续 no_break 链扩展出必须同段的单元块, 返回冲突单元与违例。"""
    lo, hi = u, u
    while lo - 1 in nb:
        lo -= 1
    while hi + 1 in nb:
        hi += 1
    a = lo - 1  # 单元 lo-1 .. hi 必须落在同一段
    reasons = segment_violations(units, a, hi, cfg)
    if reasons:
        return a, reasons
    # 块本身可行, 是与必须断开等约束组合后不可行: 回退到所在强制区间定位
    n = len(units)
    cuts = [0] + sorted(mb) + [n]
    for s, e in zip(cuts, cuts[1:]):
        if s <= u <= e:
            r = _range_conflict(units, cfg, s, e, nb)
            if r:
                return r
    return a, ["与相邻边界约束组合后无可行分段"]


def _plain_conflict(units: list[Unit], dp: list[float], cfg: Config) -> dict:
    """与边界约束无关的不可行: 定位首个无法被任何可行分段覆盖的单元区间。"""
    INF = float("inf")
    n = len(units)
    # 首个不可覆盖单元 = 最后一个可达切点之后的单元(中间不可达位置可能随后恢复)
    a = max(j for j in range(n + 1) if dp[j] != INF)
    reasons = segment_violations(units, a, a, cfg)
    if not reasons:
        # 单元单独可行, 但与前方可达边界拼接均失败: 汇总最近一段的违例
        for i in range(a - 1, -1, -1):
            if dp[i] != INF:
                reasons = segment_violations(units, i, a, cfg)
                break
        if not reasons:
            reasons = ["与相邻单元组合后无可行布局"]
    wi = units[a].word_indices
    return {
        "ok": False,
        "conflict": {
            "word_range": [wi[0], wi[-1]],
            "start": round(units[a].start, 3),
            "end": round(units[a].end, 3),
            "text": units[a].text,
            "reasons": reasons,
        },
    }
