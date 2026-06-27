"""推奨候補の判定と推奨モード別スコア計算。"""

from __future__ import annotations

import pandas as pd


STABLE_MODE = "2倍回収安定モード"
EV_MODE = "期待値重視モード"
HIT_RATE_MODE = "的中率重視モード"
RECOMMENDATION_MODES = [STABLE_MODE, EV_MODE, HIT_RATE_MODE]

# 推奨モードごとの「上位から買う」並び順の基準列。
# 期待値重視のみ期待値順、それ以外は人気（推定的中確率）順を上位とする。
MODE_ORDER_COLUMN = {
    STABLE_MODE: "probability_percent",
    HIT_RATE_MODE: "probability_percent",
    EV_MODE: "expected_value",
}


def calculate_score(row: pd.Series | dict, target_payout: float, mode: str) -> float:
    """推奨モードに応じたスコアを計算する。"""

    expected_value = float(row.get("expected_value", 0.0))
    hit_probability = float(row.get("probability_percent", 0.0)) / 100.0
    payout = float(row.get("payout", 0.0))
    payout_deviation_ratio = abs(payout - float(target_payout)) / float(target_payout)

    if mode == STABLE_MODE:
        return (
            expected_value
            + float(target_payout) * hit_probability
            - float(target_payout) * payout_deviation_ratio
        )
    if mode == EV_MODE:
        return expected_value
    if mode == HIT_RATE_MODE:
        return float(target_payout) * hit_probability + expected_value * 0.3

    raise ValueError(f"未知の推奨モードです: {mode}")


def annotate_candidates(
    candidates: pd.DataFrame,
    *,
    budget: int,
    target_payout: float,
    payout_lower_ratio: float,
    payout_upper_ratio: float,
    odds_cap: float,
    positive_ev_only: bool,
    recommendation_mode: str,
) -> pd.DataFrame:
    """全候補へ判定列・スコア列を付与する。

    推奨候補条件:
        - 必要購入額が予算以内
        - 丸め後のモデル上の推定払戻額が目標払戻許容範囲内
        - 推定三連複オッズが上限以内
        - positive_ev_only=True の場合は期待値がプラス
    """

    if payout_lower_ratio >= payout_upper_ratio:
        raise ValueError("目標払戻許容下限は上限より小さい必要があります。")
    if budget <= 0:
        raise ValueError("総予算は正の整数である必要があります。")
    if odds_cap <= 0:
        raise ValueError("推定三連複オッズ上限は0より大きい必要があります。")
    if recommendation_mode not in RECOMMENDATION_MODES:
        raise ValueError(f"未知の推奨モードです: {recommendation_mode}")

    df = candidates.copy()
    lower = float(target_payout) * float(payout_lower_ratio)
    upper = float(target_payout) * float(payout_upper_ratio)

    df["hit_probability_decimal"] = pd.to_numeric(df["probability_percent"], errors="coerce") / 100.0
    df["payout_diff"] = pd.to_numeric(df["payout"], errors="coerce") - float(target_payout)
    df["payout_deviation_ratio"] = df["payout_diff"].abs() / float(target_payout)
    df["within_target_payout_range"] = df["payout"].between(lower, upper, inclusive="both")
    df["odds_cap_exceeded"] = pd.to_numeric(df["odds"], errors="coerce") > float(odds_cap)
    df["within_budget"] = pd.to_numeric(df["stake"], errors="coerce") <= int(budget)
    df["positive_ev"] = pd.to_numeric(df["expected_value"], errors="coerce") > 0

    df["score"] = df.apply(
        lambda row: calculate_score(row, target_payout=target_payout, mode=recommendation_mode),
        axis=1,
    )
    df["is_recommendation_candidate"] = (
        df["within_budget"]
        & df["within_target_payout_range"]
        & ~df["odds_cap_exceeded"]
        & (df["positive_ev"] if positive_ev_only else True)
    )

    return df


def recommendation_empty_reasons(
    annotated: pd.DataFrame,
    *,
    positive_ev_only: bool,
) -> list[str]:
    """推奨候補が0件のときの理由を生成する。"""

    reasons: list[str] = []
    if annotated.empty:
        return ["三連複候補が生成されませんでした。頭数と単勝オッズを確認してください。"]
    if not annotated["within_budget"].any():
        reasons.append("必要購入額が総予算以内に収まる買い目がありません。")
    if not annotated["within_target_payout_range"].any():
        reasons.append("モデル上の推定払戻額が目標払戻許容範囲内に入る買い目がありません。")
    if annotated["odds_cap_exceeded"].all():
        reasons.append("すべての買い目が推定三連複オッズ上限を超えています。")
    if positive_ev_only and not annotated["positive_ev"].any():
        reasons.append("期待値プラス条件を満たす買い目がありません。")

    combined = annotated[
        annotated["within_budget"]
        & annotated["within_target_payout_range"]
        & ~annotated["odds_cap_exceeded"]
        & (annotated["positive_ev"] if positive_ev_only else True)
    ]
    if combined.empty and not reasons:
        reasons.append("複数条件の組み合わせにより推奨候補が0件になっています。")
    return reasons


def allocate_full_budget(
    candidates: pd.DataFrame,
    *,
    budget: int,
    unit: int,
    max_bets: int,
    target_payout: float,
    recommendation_mode: str,
    payout_upper: float | None = None,
) -> list[dict]:
    """予算を使い切る前提で、上位の三連複から順に買い目を割り当てる。

    各買い目を目標払戻額に近い基本購入額で買うと、セット期待値は
        期待値 = 目標払戻額 × (選んだ買い目の的中確率合計) - 合計購入額
    に比例する。よって、余った予算は既存買い目への上乗せではなく、
    「まだ買っていない“2倍程度で返る”上位の買い目」を買い増す方が期待値が高い。

    手順:
        1. 推奨モードに応じた上位順（人気順または期待値順）に並べる。
        2. 予算と最大買い目数の範囲で、上位から順に基本購入額で買い増す。
           余りが出たら、その余りで買える次の上位買い目を探して買い足す。
        3. これ以上、別の買い目を基本購入額で買えない端数だけは、
           既存の上位買い目に最低購入単位ずつ上乗せして使い切る。
        4. 最終的な購入額に合わせて払戻額・期待値・スコアを再計算する。

    予算が最低購入単位で割り切れない場合のみ、単位未満の端数だけが残る。
    """

    if budget <= 0:
        raise ValueError("budget は正の整数である必要があります。")
    if unit <= 0:
        raise ValueError("unit は正の整数である必要があります。")
    if max_bets < 1:
        raise ValueError("max_bets は1以上である必要があります。")

    df = candidates.copy()
    if df.empty:
        return []

    order_column = MODE_ORDER_COLUMN.get(recommendation_mode, "probability_percent")
    if order_column not in df.columns:
        order_column = "probability_percent"
    # 上位＝人気(確率高)/期待値高 を先頭にする。同点は推定オッズ昇順で安定化。
    df = df.sort_values(
        [order_column, "odds"],
        ascending=[False, True],
    ).reset_index(drop=True)

    rows: list[dict] = []
    for _, row in df.iterrows():
        base_stake = int(round(float(row.get("stake", 0))))
        if base_stake <= 0:
            continue
        record = dict(row)
        record["base_stake"] = base_stake
        rows.append(record)

    if not rows:
        return []

    # 予算と最大買い目数の範囲で、上位から順に「別の買い目」を基本購入額で買い足す。
    # 各反復で、現在の残予算で買える最上位の未選択買い目を1点選ぶ。
    selected: list[dict] = []
    selected_keys: set[str] = set()
    total = 0
    while len(selected) < int(max_bets):
        remaining = int(budget) - total
        pick = None
        for record in rows:
            if record["bet"] in selected_keys:
                continue
            if record["base_stake"] <= remaining:
                pick = record
                break
        if pick is None:
            break
        chosen = dict(pick)
        chosen["stake"] = int(pick["base_stake"])
        selected.append(chosen)
        selected_keys.add(pick["bet"])
        total += int(pick["base_stake"])

    if not selected:
        return []

    # これ以上、別の買い目を基本購入額で買えない端数だけ、上位の買い目に上乗せして使い切る。
    # ただし、上乗せで払戻が許容上限を超えないようにする（超高配当化を防ぐ）。
    leftover = int(budget) - total
    eps = 1e-9
    while leftover >= int(unit):
        progressed = False
        for record in selected:  # 上位から順に
            if leftover < int(unit):
                break
            new_stake = int(record["stake"]) + int(unit)
            new_payout = new_stake * float(record["odds"])
            if payout_upper is not None and new_payout > float(payout_upper) + eps:
                continue
            record["stake"] = new_stake
            leftover -= int(unit)
            progressed = True
        if not progressed:
            break

    # 上乗せ後の購入額に合わせて各指標を再計算する。
    for record in selected:
        record.pop("base_stake", None)
        stake = int(record["stake"])
        odds = float(record["odds"])
        probability = float(record.get("probability_percent", 0.0)) / 100.0
        payout = stake * odds
        record["payout"] = float(payout)
        record["expected_value"] = float(probability * payout - stake)
        record["payout_diff"] = float(payout - float(target_payout))
        record["score"] = float(
            calculate_score(record, target_payout=target_payout, mode=recommendation_mode)
        )

    return selected
