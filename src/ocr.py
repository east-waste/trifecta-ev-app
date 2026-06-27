"""画像から馬番・馬名・単勝オッズを抽出するOCR補助機能。

Google Gemini API（無料枠あり）の画像入力を使う。公開GitHub Pages上で
APIキーを直接扱うのは推奨しないため、主にローカルStreamlit実行向け。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from dataclasses import dataclass

import pandas as pd

# 既定の各プロバイダのモデル名。
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

OCR_PROMPT = """
画像には競馬の出馬表・オッズ表が写っています。
馬番、馬名、単勝オッズを読み取り、次のJSON配列だけを返してください。
説明文、Markdown、コードフェンスは不要です。

[
  {"horse_number": 1, "horse_name": "馬名", "win_odds": 12.3}
]

注意:
- 単勝オッズだけを抽出してください。
- 複勝、馬連、三連複など他のオッズは無視してください。
- 読み取れない馬名は空文字にしてください。
- horse_number は整数、win_odds は小数で返してください。
"""


@dataclass
class OcrResult:
    dataframe: pd.DataFrame
    raw_text: str


def image_bytes_to_data_url(image_bytes: bytes, filename: str | None = None) -> str:
    """画像bytesをbase64 data URLに変換する。"""

    mime = "image/png"
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed and guessed.startswith("image/"):
            mime = guessed
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _extract_json_array(text: str) -> list[dict]:
    """モデル応答からJSON配列を取り出す。"""

    cleaned = text.strip()
    # ```json ... ``` に包まれた場合に対応
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 応答文中の最初の配列らしき部分を拾う
        array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, flags=re.S)
        if not array_match:
            raise
        parsed = json.loads(array_match.group(0))

    if isinstance(parsed, dict) and "horses" in parsed:
        parsed = parsed["horses"]
    if not isinstance(parsed, list):
        raise ValueError("OCR結果がJSON配列ではありません。")
    return parsed


def normalize_ocr_records(records: list[dict]) -> pd.DataFrame:
    """OCR結果を horse_number / horse_name / win_odds のDataFrameに正規化する。"""

    rows: list[dict] = []
    for record in records:
        horse_number = (
            record.get("horse_number")
            or record.get("number")
            or record.get("馬番")
            or record.get("umaban")
        )
        horse_name = (
            record.get("horse_name")
            or record.get("name")
            or record.get("馬名")
            or ""
        )
        win_odds = (
            record.get("win_odds")
            or record.get("odds")
            or record.get("単勝オッズ")
            or record.get("単勝")
        )

        if horse_number in (None, "") or win_odds in (None, ""):
            continue

        rows.append(
            {
                "horse_number": horse_number,
                "horse_name": horse_name,
                "win_odds": win_odds,
            }
        )

    df = pd.DataFrame(rows, columns=["horse_number", "horse_name", "win_odds"])
    if df.empty:
        return df

    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce")
    df["win_odds"] = (
        df["win_odds"]
        .astype(str)
        .str.replace("倍", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")
    df = df.dropna(subset=["horse_number", "win_odds"]).copy()
    df["horse_number"] = df["horse_number"].astype(int)
    df["horse_name"] = df["horse_name"].fillna("").astype(str)
    df["win_odds"] = df["win_odds"].astype(float)
    df = df.sort_values("horse_number").reset_index(drop=True)
    return df


def _guess_mime(filename: str | None) -> str:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed and guessed.startswith("image/"):
            return guessed
    return "image/png"


def _call_gemini(image_bytes: bytes, filename: str | None, api_key: str, model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - 環境依存
        raise RuntimeError(
            "google-genai パッケージがインストールされていません。"
            " `python3 -m pip install -r requirements.txt` を実行してください。"
        ) from exc

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=_guess_mime(filename))
    # 503(UNAVAILABLE)や一時的な混雑は自動で数回リトライする。
    max_attempts = 4
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[OCR_PROMPT, image_part],
            )
            return getattr(response, "text", "") or str(response)
        except Exception as exc:  # pragma: no cover - API依存
            message = str(exc)
            last_exc = exc
            is_quota = (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "Quota exceeded" in message
            )
            is_transient = (
                "503" in message
                or "UNAVAILABLE" in message
                or "high demand" in message
                or "overloaded" in message
                or "500" in message
                or "INTERNAL" in message
            )
            if is_quota:
                raise RuntimeError(
                    "Gemini APIの無料枠またはレート制限に達しています。"
                    " 少し待って再試行するか、別のGoogle AI Studio APIキー/プロジェクトを使う、"
                    " またはGoogle AI StudioのUsage/Rate limitでクォータを確認してください。"
                ) from exc
            if is_transient and attempt < max_attempts - 1:
                # 指数バックオフ（2,4,8秒）で待ってから再試行する。
                time.sleep(2 ** (attempt + 1))
                continue
            if is_transient:
                raise RuntimeError(
                    "Gemini APIが一時的に高負荷（503）です。数回再試行しましたが応答がありませんでした。"
                    " 少し時間をおいて再試行するか、モデルを gemini-2.5-flash / gemini-2.0-flash に変えて試してください。"
                ) from exc
            raise
    # ここには通常到達しない
    raise RuntimeError(f"Gemini API呼び出しに失敗しました: {last_exc}")


def extract_horse_odds_from_image(
    *,
    image_bytes: bytes,
    filename: str | None,
    api_key: str,
    provider: str = "gemini",
    model: str | None = None,
) -> OcrResult:
    """画像から馬番・馬名・単勝オッズを抽出する。

    Args:
        provider: "gemini"（現在はGeminiのみ対応）。
        model: 省略時はプロバイダ既定モデルを使う。
    """

    if not api_key:
        raise ValueError("APIキーが空です。")

    provider = (provider or "gemini").lower()
    if provider != "gemini":
        raise ValueError(
            "現在はGemini APIのみ対応しています（無料枠運用のため）。provider='gemini' を指定してください。"
        )
    used_model = model or DEFAULT_GEMINI_MODEL
    raw_text = _call_gemini(image_bytes, filename, api_key, used_model)

    records = _extract_json_array(raw_text)
    df = normalize_ocr_records(records)
    if df.empty:
        raise ValueError("画像から馬番・単勝オッズを抽出できませんでした。")
    return OcrResult(dataframe=df, raw_text=raw_text)
