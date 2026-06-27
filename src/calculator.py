"""三連複の購入額・払戻額・期待値計算ユーティリティ。"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def ceil_to_unit(value: float, unit: int) -> int:
    """value を unit 単位に切り上げる。

    Args:
        value: 切り上げ対象の金額。
        unit: 最低購入単位。正の整数である必要がある。

    Returns:
        unit 単位へ切り上げた整数金額。
    """

    if unit <= 0:
        raise ValueError("unit は正の整数である必要があります。")
    if value <= 0:
        return 0
    return int(math.ceil(float(value) / unit) * unit)


def calculate_required_stake(target_payout: float, odds: float, unit: int) -> int:
    """目標払戻額を満たすための購入額を計算する。"""

    if odds <= 0:
        raise ValueError("odds は正の値である必要があります。")
    return ceil_to_unit(float(target_payout) / float(odds), unit)


def calculate_theoretical_stake(target_payout: float, odds: float) -> float:
    """丸め前の理論購入額を計算する。"""

    if odds <= 0:
        raise ValueError("odds は正の値である必要があります。")
    return float(target_payout) / float(odds)


def calculate_payout(stake: float, odds: float) -> float:
    """購入額とオッズから払戻額を計算する。"""

    return float(stake) * float(odds)


def probability_percent_to_decimal(probability_percent: float) -> float:
    """パーセント表記の的中確率を 0〜1 の小数へ変換する。"""

    return float(probability_percent) / 100.0


def calculate_expected_value(
    probability_percent: float,
    payout: float,
    stake: float,
) -> float:
    """買い目単位の期待値を計算する。

    期待値 = 的中確率(小数) × 払戻額 - 購入額
    """

    p = probability_percent_to_decimal(probability_percent)
    return p * float(payout) - float(stake)


def enrich_candidates(
    df: pd.DataFrame,
    target_payout: float,
    unit: int,
) -> pd.DataFrame:
    """入力 DataFrame に購入額・払戻額・期待値列を追加する。

    想定列:
        bet, odds, probability_percent, memo
    """

    rows: list[dict] = []
    for _, row in df.iterrows():
        odds = float(row["odds"])
        probability_percent = float(row["probability_percent"])
        theoretical_stake = calculate_theoretical_stake(target_payout, odds)
        stake = calculate_required_stake(target_payout, odds, unit)
        payout = calculate_payout(stake, odds)
        expected_value = calculate_expected_value(probability_percent, payout, stake)

        rows.append(
            {
                "bet": str(row["bet"]),
                "odds": odds,
                "probability_percent": probability_percent,
                "market_probability_percent": float(row["market_probability_percent"])
                if "market_probability_percent" in row and not pd.isna(row.get("market_probability_percent"))
                else None,
                "memo": "" if pd.isna(row.get("memo", "")) else str(row.get("memo", "")),
                "theoretical_stake": float(theoretical_stake),
                "stake": int(stake),
                "payout": float(payout),
                "payout_diff": float(payout - float(target_payout)),
                "expected_value": float(expected_value),
            }
        )

    return pd.DataFrame(rows)


def summarize_bet_set(selected_bets: Iterable[dict]) -> dict:
    """選択された買い目セット全体の指標を集計する。"""

    bets = list(selected_bets)
    total_stake = sum(float(bet.get("stake", 0)) for bet in bets)
    expected_return = sum(
        probability_percent_to_decimal(float(bet.get("probability_percent", 0)))
        * float(bet.get("payout", 0))
        for bet in bets
    )
    hit_probability = sum(float(bet.get("probability_percent", 0)) for bet in bets)
    payouts = [float(bet.get("payout", 0)) for bet in bets]

    return {
        "selected_bets": bets,
        "total_stake": int(total_stake),
        "expected_return": float(expected_return),
        "expected_value": float(expected_return - total_stake),
        "hit_probability": float(hit_probability),
        "min_payout": float(min(payouts)) if payouts else 0.0,
        "max_payout": float(max(payouts)) if payouts else 0.0,
        "average_payout": float(sum(payouts) / len(payouts)) if payouts else 0.0,
    }
