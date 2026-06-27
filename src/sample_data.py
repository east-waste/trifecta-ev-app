"""Streamlit 初期表示用サンプルデータ。"""

from __future__ import annotations

import pandas as pd


def get_sample_data() -> pd.DataFrame:
    """サンプル買い目データを返す。"""

    return pd.DataFrame(
        [
            {"bet": "1-2-3", "odds": 10.0, "probability_percent": 15.0, "memo": "本命寄り"},
            {"bet": "1-2-4", "odds": 18.0, "probability_percent": 8.0, "memo": "中穴"},
            {"bet": "1-3-5", "odds": 35.0, "probability_percent": 4.0, "memo": "穴"},
            {"bet": "2-4-7", "odds": 80.0, "probability_percent": 1.0, "memo": "大穴"},
            {"bet": "1-5-8", "odds": 55.0, "probability_percent": 2.5, "memo": "抑え"},
            {"bet": "3-6-9", "odds": 120.0, "probability_percent": 0.8, "memo": "超穴"},
        ]
    )
