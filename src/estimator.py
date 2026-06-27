"""単勝オッズから三連複の的中確率・オッズを推定する（Harville法）。"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Mapping

import pandas as pd

# 三連複オッズを推定するときの控除率の初期値。
# 実際の控除率は主催者・券種・時期で異なるため、アプリ側で調整できるようにしている。
DEFAULT_TRIFECTA_TAKEOUT = 0.25


def win_odds_to_win_probabilities(win_odds: Mapping[int, float]) -> dict[int, float]:
    """単勝オッズを正規化した1着確率へ変換する。

    各馬の暫定勝率 = 1 / 単勝オッズ。
    控除率が含まれ合計が1を超えるため、全体で正規化する。
    """

    inverse = {
        horse: 1.0 / float(odds)
        for horse, odds in win_odds.items()
        if odds is not None and float(odds) > 0
    }
    total = sum(inverse.values())
    if total <= 0:
        raise ValueError("有効な単勝オッズが1件もありません。")
    return {horse: value / total for horse, value in inverse.items()}


def harville_trifecta_probabilities(
    win_probabilities: Mapping[int, float],
    lam: float = 1.0,
) -> dict[tuple[int, int, int], float]:
    """1着確率から三連複（順不同3頭）の的中確率を計算する。

    Harville法では i→j→k の順で 1,2,3着になる確率を
        P = p_i × p_j/(1-p_i) × p_k/(1-p_i-p_j)
    として連鎖計算する。lam は2・3着での人気馬偏重を補正する係数で、
    lam < 1 にすると人気薄の確率を相対的に引き上げる。
    """

    horses = list(win_probabilities.keys())
    p = {h: float(win_probabilities[h]) for h in horses}
    result: dict[tuple[int, int, int], float] = {}

    for combo in combinations(horses, 3):
        total = 0.0
        for i, j, k in permutations(combo, 3):
            p1 = p[i]
            denom2 = sum(p[m] ** lam for m in horses if m != i)
            p2 = (p[j] ** lam / denom2) if denom2 > 0 else 0.0
            denom3 = sum(p[m] ** lam for m in horses if m not in (i, j))
            p3 = (p[k] ** lam / denom3) if denom3 > 0 else 0.0
            total += p1 * p2 * p3
        result[tuple(sorted(combo))] = total

    return result


def estimate_trifecta_odds(probability: float, takeout: float = DEFAULT_TRIFECTA_TAKEOUT) -> float:
    """的中確率（0〜1）から三連複オッズの目安を推定する。

    オッズ ≒ (1 - 控除率) / 的中確率
    """

    if probability <= 0:
        return float("inf")
    return (1.0 - float(takeout)) / float(probability)


def build_trifecta_candidates(
    win_odds: Mapping[int, float],
    belief_lambda: float = 0.80,
    market_lambda: float = 1.0,
    takeout: float = DEFAULT_TRIFECTA_TAKEOUT,
) -> pd.DataFrame:
    """単勝オッズから全三連複の候補（買い目・推定オッズ・推定確率）を生成する。

    期待値がプラスになり得る買い目を見つけるには、「自分が信じる真の確率」と
    「市場が三連複を価格づけする確率」がズレている必要がある。

    本関数では2つの Harville モデルを使う。
        belief: 真の確率の推定（Henery型補正。belief_lambda < 1 で人気薄の連対を高めに見積もる）
        market: 市場の価格づけの推定（素朴な Harville。通常 market_lambda = 1.0）

    推定的中確率(probability_percent) は belief 側、
    推定三連複オッズ(odds) は market 側（控除率込み）から計算する。
    belief_lambda == market_lambda のときは両者が一致し、
    全買い目の期待値は -控除率×購入額 となる（＝エッジなし）。

    Returns:
        bet, odds, probability_percent, memo 列を持つ DataFrame。
        probability_percent の降順に並ぶ。
    """

    win_probabilities = win_odds_to_win_probabilities(win_odds)
    belief = harville_trifecta_probabilities(win_probabilities, lam=belief_lambda)
    market = harville_trifecta_probabilities(win_probabilities, lam=market_lambda)

    rows: list[dict] = []
    for combo, belief_probability in belief.items():
        market_probability = market[combo]
        odds = estimate_trifecta_odds(market_probability, takeout=takeout)
        bet = "-".join(str(h) for h in combo)
        rows.append(
            {
                "bet": bet,
                "odds": round(float(odds), 1),
                "probability_percent": round(float(belief_probability) * 100.0, 4),
                "market_probability_percent": round(float(market_probability) * 100.0, 4),
                "memo": "単勝オッズから推定",
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["bet", "odds", "probability_percent", "market_probability_percent", "memo"],
    )
    if df.empty:
        return df
    # 表として自然に見えるようにオッズ昇順（＝人気順）で並べる。
    return df.sort_values("odds", ascending=True).reset_index(drop=True)
