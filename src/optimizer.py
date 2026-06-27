"""制約を満たす候補からスコア最大の買い目セットを選ぶ。"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .calculator import summarize_bet_set


def _to_records(candidates: Iterable[dict] | pd.DataFrame) -> list[dict]:
    if isinstance(candidates, pd.DataFrame):
        return candidates.to_dict(orient="records")
    return [dict(candidate) for candidate in candidates]


def _stake_gcd(records: list[dict], budget: int) -> int:
    """DP の重さ圧縮に使う購入額の最大公約数を求める。"""

    values = [int(round(float(r.get("stake", 0)))) for r in records if float(r.get("stake", 0)) > 0]
    values.append(int(budget))
    gcd_value = 0
    for value in values:
        gcd_value = math.gcd(gcd_value, abs(value))
    return max(gcd_value, 1)


def _is_better(
    candidate_value: float,
    candidate_stake: int,
    candidate_hit_probability: float,
    current_value: float,
    current_stake: int,
    current_hit_probability: float,
) -> bool:
    """DP の同点比較。

    第一優先: 期待値が大きい
    第二優先: 購入額が小さい
    第三優先: 推定的中確率が大きい
    """

    eps = 1e-9
    if candidate_value > current_value + eps:
        return True
    if abs(candidate_value - current_value) <= eps:
        if candidate_stake < current_stake:
            return True
        if candidate_stake == current_stake and candidate_hit_probability > current_hit_probability + eps:
            return True
    return False


def optimize_bets(
    candidates: Iterable[dict] | pd.DataFrame,
    budget: int,
    max_bets: int,
    positive_ev_only: bool = True,
    value_column: str = "expected_value",
) -> dict:
    """候補から指定 value_column の合計が最大となる買い目セットを選ぶ。

    Args:
        candidates: stake, payout, expected_value, probability_percent を持つ候補。
        budget: 総予算。
        max_bets: 選択できる最大買い目数。
        positive_ev_only: True の場合は期待値が 0 以下の買い目を除外する。
        value_column: DPで最大化する列名。従来互換の初期値は expected_value。

    Returns:
        selected_bets, total_stake, expected_return, expected_value,
        hit_probability, min_payout, max_payout を含む辞書。
    """

    if budget <= 0:
        raise ValueError("budget は正の整数である必要があります。")
    if max_bets < 1:
        raise ValueError("max_bets は1以上である必要があります。")

    records = _to_records(candidates)
    filtered: list[dict] = []
    for record in records:
        stake = int(round(float(record.get("stake", 0))))
        expected_value = float(record.get("expected_value", 0))
        if stake <= 0 or stake > budget:
            continue
        if positive_ev_only and expected_value <= 0:
            continue
        copied = dict(record)
        copied["stake"] = stake
        filtered.append(copied)

    if not filtered:
        return summarize_bet_set([])

    unit = _stake_gcd(filtered, budget)
    capacity = int(budget // unit)

    # dp[k][w] = (期待値, 合計購入額, 的中確率合計, 選択index tuple)
    dp: list[list[tuple[float, int, float, tuple[int, ...]] | None]] = [
        [None for _ in range(capacity + 1)] for _ in range(max_bets + 1)
    ]
    dp[0][0] = (0.0, 0, 0.0, tuple())

    for idx, record in enumerate(filtered):
        stake = int(record["stake"])
        weight = int(stake // unit)
        value = float(record.get(value_column, record.get("expected_value", 0)))
        hit_probability = float(record.get("probability_percent", 0))

        for k in range(max_bets - 1, -1, -1):
            for w in range(capacity - weight, -1, -1):
                state = dp[k][w]
                if state is None:
                    continue
                prev_value, prev_stake, prev_hit, prev_indices = state
                new_k = k + 1
                new_w = w + weight
                new_value = prev_value + value
                new_stake = prev_stake + stake
                new_hit = prev_hit + hit_probability
                current = dp[new_k][new_w]

                if current is None or _is_better(
                    new_value,
                    new_stake,
                    new_hit,
                    current[0],
                    current[1],
                    current[2],
                ):
                    dp[new_k][new_w] = (new_value, new_stake, new_hit, prev_indices + (idx,))

    # 候補が存在する場合は、できるだけ非空の組み合わせを推奨する。
    # （スコアが負になりうるモードでも、範囲内候補があれば買い目を提示する）
    best: tuple[float, int, float, tuple[int, ...]] | None = None
    for k in range(1, max_bets + 1):
        for w in range(capacity + 1):
            state = dp[k][w]
            if state is None:
                continue
            if best is None or _is_better(
                state[0], state[1], state[2], best[0], best[1], best[2]
            ):
                best = state

    if best is None:
        return summarize_bet_set([])

    selected = [filtered[i] for i in best[3]]
    return summarize_bet_set(selected)
