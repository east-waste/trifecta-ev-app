"""入力データと設定値のバリデーション。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = ["bet", "odds", "probability_percent", "memo"]

JAPANESE_COLUMN_MAP = {
    "買い目": "bet",
    "三連複オッズ": "odds",
    "的中確率（%）": "probability_percent",
    "的中確率(%)": "probability_percent",
    "的中確率": "probability_percent",
    "メモ": "memo",
}

ENGLISH_COLUMN_MAP = {
    "bet": "bet",
    "odds": "odds",
    "probability_percent": "probability_percent",
    "memo": "memo",
}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


def normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """英語列名・日本語列名のCSVを内部列名へ正規化する。"""

    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    normalized = df.copy()
    rename_map: dict[str, str] = {}
    for column in normalized.columns:
        stripped = str(column).strip()
        if stripped in JAPANESE_COLUMN_MAP:
            rename_map[column] = JAPANESE_COLUMN_MAP[stripped]
        elif stripped in ENGLISH_COLUMN_MAP:
            rename_map[column] = ENGLISH_COLUMN_MAP[stripped]

    normalized = normalized.rename(columns=rename_map)

    for column in REQUIRED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "" if column in {"bet", "memo"} else 0.0

    normalized = normalized[REQUIRED_COLUMNS].copy()
    normalized["bet"] = normalized["bet"].fillna("").astype(str).str.strip()
    normalized["memo"] = normalized["memo"].fillna("").astype(str)
    normalized["odds"] = pd.to_numeric(normalized["odds"], errors="coerce")
    normalized["probability_percent"] = pd.to_numeric(
        normalized["probability_percent"],
        errors="coerce",
    )
    return normalized


def validate_settings(
    budget: int,
    target_multiplier: float,
    unit: int,
    max_bets: int,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if int(budget) <= 0:
        errors.append("総予算は正の整数で入力してください。")
    if float(target_multiplier) <= 0:
        errors.append("目標回収倍率は正の数で入力してください。")
    if int(unit) <= 0:
        errors.append("最低購入単位は正の整数で入力してください。")
    if int(max_bets) < 1:
        errors.append("最大買い目数は1以上で入力してください。")

    return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


def validate_bet_dataframe(df: pd.DataFrame) -> ValidationResult:
    """買い目入力の検証を行う。"""

    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        errors.append("買い目データを1件以上入力してください。")
        return ValidationResult(False, errors, warnings)

    duplicated_bets = sorted(
        {
            bet
            for bet in df.loc[df["bet"].duplicated(keep=False), "bet"].tolist()
            if str(bet).strip()
        }
    )

    for idx, row in df.iterrows():
        line = int(idx) + 1
        bet = str(row.get("bet", "")).strip()
        odds = row.get("odds")
        probability = row.get("probability_percent")

        if not bet:
            errors.append(f"{line}行目: 買い目が空です。")

        if pd.isna(odds):
            errors.append(f"{line}行目: 三連複オッズを数値で入力してください。")
        elif float(odds) <= 1.0:
            errors.append(f"{line}行目: オッズは1.0より大きい値で入力してください。")

        if pd.isna(probability):
            errors.append(f"{line}行目: 的中確率（%）を数値で入力してください。")
        elif float(probability) < 0 or float(probability) > 100:
            errors.append(f"{line}行目: 的中確率（%）は0以上100以下で入力してください。")

    if duplicated_bets:
        errors.append("買い目が重複しています: " + ", ".join(duplicated_bets))

    probability_sum = pd.to_numeric(df["probability_percent"], errors="coerce").fillna(0).sum()
    if probability_sum > 100:
        warnings.append(
            f"入力された的中確率の合計が100%を超えています（合計: {probability_sum:.2f}%）。"
        )

    return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)
