from __future__ import annotations

import pandas as pd
import streamlit as st

from src.calculator import enrich_candidates, summarize_bet_set
from src.estimator import DEFAULT_TRIFECTA_TAKEOUT, build_trifecta_candidates
from src.ocr import extract_horse_odds_from_image
from src.recommendation import (
    EV_MODE,
    HIT_RATE_MODE,
    RECOMMENDATION_MODES,
    STABLE_MODE,
    allocate_full_budget,
    annotate_candidates,
    recommendation_empty_reasons,
)


APP_TITLE = "三連複 2倍回収・期待値最大化シミュレーター"
DISCLAIMER = (
    "本ツールは、入力された単勝オッズと推定モデルに基づいて、三連複の買い目と資金配分を計算するシミュレーターです。"
    "表示される三連複オッズや払戻額は実際のものではなく、モデル上の推定値です。"
    "的中や利益を保証するものではありません。実際の投票は自己責任で行ってください。"
)
MODEL_NOTICE = (
    "このアプリで表示される三連複オッズや払戻額は、単勝オッズから簡易モデルで推定した値です。"
    "実際のJRA等の三連複オッズ・払戻額とは異なる可能性があります。"
)


def format_yen(value: float) -> str:
    return f"{value:,.0f}円"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def bool_to_japanese(value: bool) -> str:
    return "はい" if bool(value) else "いいえ"


def default_win_odds(num_horses: int) -> pd.DataFrame:
    """頭数に応じた単勝オッズの初期表を作る。"""

    # 初期値（10頭立てのサンプル単勝オッズ）。ユーザーがすぐ編集できる。
    odds = [
        33.9, 18.0, 59.5, 7.9, 10.5, 3.1, 55.2, 148.7, 122.8, 1.7,
        45.0, 60.0, 80.0, 100.0, 130.0, 160.0, 200.0, 250.0,
    ]
    df = pd.DataFrame(
        {
            "horse_number": list(range(1, num_horses + 1)),
            "horse_name": ["" for _ in range(num_horses)],
            "win_odds": odds[:num_horses],
        }
    )
    # 小数第一位まで入力できるよう float 型を明示する（int になると小数が入らない）。
    df["horse_number"] = df["horse_number"].astype(int)
    df["win_odds"] = df["win_odds"].astype(float)
    return df


def build_win_odds_df(num_horses: int, previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """指定頭数の単勝オッズ表を作る。

    既存の入力(previous)があれば、馬番が一致する分の単勝オッズを引き継ぐ。
    これにより、頭数を変えて作り直しても入力済みの値が無駄に消えない。
    """

    df = default_win_odds(num_horses)
    if previous is not None and not previous.empty and "win_odds" in previous.columns:
        prev_numbers = pd.to_numeric(previous["horse_number"], errors="coerce")
        prev_odds = pd.to_numeric(previous["win_odds"], errors="coerce")
        prev_names = previous["horse_name"] if "horse_name" in previous.columns else ["" for _ in range(len(previous))]
        prev_map = {
            int(n): {"win_odds": float(o), "horse_name": str(name)}
            for n, o, name in zip(prev_numbers, prev_odds, prev_names)
            if pd.notna(n) and pd.notna(o)
        }
        df["win_odds"] = [
            prev_map.get(int(h), {"win_odds": float(default)})["win_odds"]
            for h, default in zip(df["horse_number"], df["win_odds"])
        ]
        df["horse_name"] = [
            prev_map.get(int(h), {"horse_name": ""})["horse_name"]
            for h in df["horse_number"]
        ]
        df["win_odds"] = df["win_odds"].astype(float)
    return df


def render_win_odds_editor() -> tuple[pd.DataFrame, bool]:
    """確定済みの単勝オッズ表を form 内で編集する。

    st.data_editor は通常セル編集時に即 rerun されるため、form に入れて
    ボタン押下時だけ値を反映・計算する。頭数の確定後にのみ呼ばれる。
    """

    locked_num_horses = int(st.session_state["locked_num_horses"])

    with st.form("win_odds_form", clear_on_submit=False):
        edited = st.data_editor(
            st.session_state.win_odds_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "horse_number": st.column_config.NumberColumn(
                    "馬番",
                    disabled=True,
                    format="%d",
                ),
                "horse_name": st.column_config.TextColumn(
                    "馬名",
                    help="画像OCRで読み取った馬名。手入力・修正もできます。",
                ),
                "win_odds": st.column_config.NumberColumn(
                    "単勝オッズ",
                    min_value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="小数第一位まで入力できます（例: 32.9）。",
                    required=True,
                ),
            },
            # 確定した頭数ごとに別ウィジェット扱いにし、古い編集状態を引きずらない。
            key=f"win_odds_editor_{locked_num_horses}",
        )
        submitted = st.form_submit_button(
            "単勝オッズを反映して三連複を予想・買い方を計算する",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        edited = edited.copy()
        edited["horse_number"] = pd.to_numeric(edited["horse_number"], errors="coerce").astype(int)
        if "horse_name" not in edited.columns:
            edited["horse_name"] = ""
        edited["horse_name"] = edited["horse_name"].fillna("").astype(str)
        edited["win_odds"] = pd.to_numeric(edited["win_odds"], errors="coerce")
        st.session_state.win_odds_df = edited
        return edited, True

    return st.session_state.win_odds_df, False


def validate_win_odds(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if df.empty:
        errors.append("単勝オッズを入力してください。")
        return errors

    if df["win_odds"].isna().any():
        errors.append("単勝オッズが空の馬があります。")
    invalid = df[df["win_odds"] <= 1.0]
    if not invalid.empty:
        numbers = ", ".join(str(int(n)) for n in invalid["horse_number"].tolist())
        errors.append(f"単勝オッズは1.0より大きい値で入力してください。対象馬番: {numbers}")
    return errors


def validate_settings(
    *,
    num_horses: int,
    budget: int,
    unit: int,
    max_bets: int,
    payout_lower_ratio: float,
    payout_upper_ratio: float,
    odds_cap: float,
) -> list[str]:
    errors: list[str] = []
    if int(num_horses) < 3:
        errors.append("頭数は3以上で入力してください。")
    if int(budget) <= 0:
        errors.append("総予算は正の整数で入力してください。")
    if int(unit) <= 0:
        errors.append("最低購入単位は正の整数で入力してください。")
    if int(max_bets) < 1:
        errors.append("最大買い目数は1以上で入力してください。")
    if float(payout_lower_ratio) >= float(payout_upper_ratio):
        errors.append("目標払戻許容下限は、目標払戻許容上限より小さい値にしてください。")
    if float(odds_cap) <= 0:
        errors.append("推定三連複オッズ上限は0より大きい値にしてください。")
    return errors


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🐎", layout="wide")
    st.title(APP_TITLE)
    st.info(DISCLAIMER)
    st.warning(MODEL_NOTICE)

    st.markdown(
        """
        この版では、**CSVで三連複オッズを入力する方式ではなく**、
        まず頭数を選び、各馬番の**単勝オッズ**を入力します。
        そこから Harville 法を使って、全三連複の**推定的中確率**と**推定三連複オッズ**を自動生成します。
        「自分が信じる確率」と「市場の三連複価格づけ」にズレがあるという仮定のもとで、
        **2倍以上なら何でもOKではなく、2倍程度に近いモデル上の推定払戻**を狙います。
        """
    )

    st.sidebar.header("1. 条件設定")
    default_config = st.session_state.get(
        "locked_config",
        {
            "num_horses": 10,
            "budget": 10000,
            "target_multiplier": 2.0,
            "unit": 100,
            "max_bets": 10,
            "positive_ev_only": False,
            "recommendation_mode": STABLE_MODE,
            "payout_lower_ratio": 0.9,
            "payout_upper_ratio": 1.3,
            "odds_cap": 200.0,
            "belief_lambda": 0.80,
            "market_lambda": 1.00,
            "takeout": float(DEFAULT_TRIFECTA_TAKEOUT),
        },
    )
    defaults = {
        "num_horses": 10,
        "budget": 10000,
        "target_multiplier": 2.0,
        "unit": 100,
        "max_bets": 10,
        "positive_ev_only": False,
        "recommendation_mode": STABLE_MODE,
        "payout_lower_ratio": 0.9,
        "payout_upper_ratio": 1.3,
        "odds_cap": 200.0,
        "belief_lambda": 0.80,
        "market_lambda": 1.00,
        "takeout": float(DEFAULT_TRIFECTA_TAKEOUT),
    }
    default_config = {**defaults, **default_config}

    with st.sidebar.form("settings_form", clear_on_submit=False):
        num_horses_input = st.number_input(
            "何頭立て",
            min_value=3,
            max_value=18,
            value=int(default_config["num_horses"]),
            step=1,
            help="ここで変更しても、下の確定ボタンを押すまで入力表は作り直されません。",
        )
        budget_input = st.number_input(
            "総予算（円）",
            min_value=1,
            value=int(default_config["budget"]),
            step=100,
        )
        target_multiplier_input = st.number_input(
            "目標回収倍率",
            min_value=0.1,
            value=float(default_config["target_multiplier"]),
            step=0.1,
            format="%.1f",
        )
        unit_input = st.number_input(
            "最低購入単位（円）",
            min_value=1,
            value=int(default_config["unit"]),
            step=100,
        )
        max_bets_input = st.number_input(
            "最大買い目数",
            min_value=1,
            value=int(default_config["max_bets"]),
            step=1,
        )
        positive_ev_only_input = st.checkbox(
            "期待値がプラスの買い目のみ推奨する",
            value=bool(default_config["positive_ev_only"]),
        )
        recommendation_mode_input = st.selectbox(
            "推奨モード",
            options=RECOMMENDATION_MODES,
            index=RECOMMENDATION_MODES.index(default_config.get("recommendation_mode", STABLE_MODE))
            if default_config.get("recommendation_mode", STABLE_MODE) in RECOMMENDATION_MODES
            else 0,
            help=(
                "2倍回収安定モード: 目標払戻額への近さ・的中確率・期待値をバランス評価。"
                "期待値重視モード: 期待値を重視。的中率重視モード: 推定的中確率を重視。"
            ),
        )
        payout_lower_ratio_input = st.number_input(
            "目標払戻許容下限",
            min_value=0.5,
            max_value=1.0,
            value=float(default_config["payout_lower_ratio"]),
            step=0.05,
            format="%.2f",
            help="目標払戻額に対する下限倍率。例: 0.9なら目標の90%以上。",
        )
        payout_upper_ratio_input = st.number_input(
            "目標払戻許容上限",
            min_value=1.0,
            max_value=3.0,
            value=float(default_config["payout_upper_ratio"]),
            step=0.05,
            format="%.2f",
            help="目標払戻額に対する上限倍率。例: 1.3なら目標の130%以下。",
        )
        odds_cap_input = st.number_input(
            "推定三連複オッズ上限",
            min_value=10.0,
            max_value=10000.0,
            value=float(default_config["odds_cap"]),
            step=10.0,
            format="%.1f",
            help="この値を超える推定三連複オッズは、初期状態では推奨候補から除外します。",
        )

        st.subheader("推定パラメータ")
        belief_lambda_input = st.slider(
            "自分の予想の人気寄せ具合",
            min_value=0.50,
            max_value=1.50,
            value=float(default_config["belief_lambda"]),
            step=0.05,
            help="値を大きくすると人気馬をより高く評価します。値を小さくすると穴馬にも確率を分散します。",
        )
        market_lambda_input = st.slider(
            "市場オッズ推定の人気寄せ具合",
            min_value=0.50,
            max_value=1.50,
            value=float(default_config["market_lambda"]),
            step=0.05,
            help="値を大きくすると人気馬をより高く評価します。値を小さくすると穴馬にも確率を分散します。",
        )
        st.caption(
            "「自分の予想の人気寄せ具合」と「市場オッズ推定の人気寄せ具合」に差をつけることで、"
            "モデル上で割安とみなされる三連複を探します。"
        )
        takeout_input = st.slider(
            "三連複 控除率の目安",
            min_value=0.00,
            max_value=0.40,
            value=float(default_config["takeout"]),
            step=0.005,
            format="%.3f",
            help="控除率は、オッズを推定するときに市場全体から差し引かれる割合の目安です。値を大きくすると推定三連複オッズは低くなります。",
        )
        st.caption(
            "控除率は、オッズを推定するときに市場全体から差し引かれる割合の目安です。"
            "値を大きくすると推定三連複オッズは低くなります。"
        )
        settings_submitted = st.form_submit_button(
            "この条件で入力表を作成・固定する",
            type="primary",
            use_container_width=True,
        )

    if settings_submitted or "locked_config" not in st.session_state:
        previous_df = st.session_state.get("win_odds_df")
        st.session_state.locked_config = {
            "num_horses": int(num_horses_input),
            "budget": int(budget_input),
            "target_multiplier": float(target_multiplier_input),
            "unit": int(unit_input),
            "max_bets": int(max_bets_input),
            "positive_ev_only": bool(positive_ev_only_input),
            "recommendation_mode": str(recommendation_mode_input),
            "payout_lower_ratio": float(payout_lower_ratio_input),
            "payout_upper_ratio": float(payout_upper_ratio_input),
            "odds_cap": float(odds_cap_input),
            "belief_lambda": float(belief_lambda_input),
            "market_lambda": float(market_lambda_input),
            "takeout": float(takeout_input),
        }
        st.session_state.locked_num_horses = int(num_horses_input)
        st.session_state.win_odds_df = build_win_odds_df(
            int(num_horses_input),
            previous=previous_df,
        )

    config = st.session_state.locked_config
    num_horses = int(config["num_horses"])
    budget = int(config["budget"])
    target_multiplier = float(config["target_multiplier"])
    unit = int(config["unit"])
    max_bets = int(config["max_bets"])
    positive_ev_only = bool(config["positive_ev_only"])
    recommendation_mode = str(config["recommendation_mode"])
    payout_lower_ratio = float(config["payout_lower_ratio"])
    payout_upper_ratio = float(config["payout_upper_ratio"])
    odds_cap = float(config["odds_cap"])
    belief_lambda = float(config["belief_lambda"])
    market_lambda = float(config["market_lambda"])
    takeout = float(config["takeout"])

    target_payout = float(budget) * float(target_multiplier)
    payout_lower = target_payout * payout_lower_ratio
    payout_upper = target_payout * payout_upper_ratio
    st.sidebar.metric("固定中の目標払戻額", format_yen(target_payout))
    st.sidebar.caption(f"許容推定払戻範囲: {format_yen(payout_lower)}〜{format_yen(payout_upper)}")

    st.header("2. 単勝オッズ入力")
    st.success(
        f"入力表は {num_horses}頭立てで固定中です。"
        "頭数や条件を変える場合は、左側で設定してから「この条件で入力表を作成・固定する」を押してください。"
    )

    with st.expander("画像から馬番・馬名・単勝オッズを自動入力する（任意）", expanded=False):
        st.caption(
            "出馬表や単勝オッズ表のスクリーンショットをアップロードすると、Gemini API等で馬番・馬名・単勝オッズを読み取り、下の入力表へ反映します。"
            "公開GitHub Pages上でAPIキーを使うのは推奨しません。ローカルStreamlit実行で使ってください。"
        )
        ocr_provider = "gemini"
        default_ocr_model = "gemini-2.5-flash-lite"
        ocr_api_key = st.text_input(
            "Gemini APIキー",
            type="password",
            help="Google AI Studio（https://aistudio.google.com/）で取得した無料枠のAPIキーを入力してください。",
        )
        ocr_model = st.text_input(
            "画像読み取りモデル",
            value=default_ocr_model,
            help="通常は初期値（gemini-2.5-flash-lite）のままでOKです。",
        )
        uploaded_odds_image = st.file_uploader(
            "馬番・馬名・単勝オッズが写った画像",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if st.button("画像から単勝オッズ表を作成する", use_container_width=True):
            if uploaded_odds_image is None:
                st.error("画像をアップロードしてください。")
            elif not ocr_api_key:
                st.error("Gemini APIキーを入力してください。")
            else:
                try:
                    ocr_result = extract_horse_odds_from_image(
                        image_bytes=uploaded_odds_image.getvalue(),
                        filename=uploaded_odds_image.name,
                        api_key=ocr_api_key,
                        provider=ocr_provider,
                        model=ocr_model.strip() or default_ocr_model,
                    )
                    ocr_df = ocr_result.dataframe
                    if ocr_df.empty:
                        st.error("画像から単勝オッズを読み取れませんでした。画像を拡大・トリミングして再試行してください。")
                    else:
                        st.session_state.win_odds_df = ocr_df[["horse_number", "horse_name", "win_odds"]].copy()
                        st.session_state.locked_num_horses = int(len(ocr_df))
                        st.session_state.locked_config["num_horses"] = int(len(ocr_df))
                        st.success(f"{len(ocr_df)}頭分の単勝オッズを読み取りました。入力表に反映します。")
                        st.dataframe(ocr_df, use_container_width=True, hide_index=True)
                        st.rerun()
                except Exception as exc:
                    st.error(f"画像読み取りに失敗しました: {exc}")

    win_odds_df, calculate_clicked = render_win_odds_editor()

    st.caption(
        "単勝オッズから1着確率を正規化し、Harville法で3着内の組み合わせ確率を推定します。"
        "自分の予想と市場オッズ推定の人気寄せ具合に差をつけることで、単勝オッズだけから「モデル上で割安そうな三連複」を探します。"
        "表示される推定三連複オッズ・モデル上の推定払戻額は、実際のオッズや払戻額ではありません。"
    )

    if not calculate_clicked:
        return

    errors = validate_win_odds(win_odds_df)
    errors.extend(
        validate_settings(
            num_horses=num_horses,
            budget=budget,
            unit=unit,
            max_bets=max_bets,
            payout_lower_ratio=payout_lower_ratio,
            payout_upper_ratio=payout_upper_ratio,
            odds_cap=odds_cap,
        )
    )

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    win_odds = {
        int(row["horse_number"]): float(row["win_odds"])
        for _, row in win_odds_df.iterrows()
    }

    raw_candidates = build_trifecta_candidates(
        win_odds,
        belief_lambda=float(belief_lambda),
        market_lambda=float(market_lambda),
        takeout=float(takeout),
    )
    enriched = enrich_candidates(raw_candidates, target_payout=target_payout, unit=int(unit))
    annotated = annotate_candidates(
        enriched,
        budget=budget,
        target_payout=target_payout,
        payout_lower_ratio=payout_lower_ratio,
        payout_upper_ratio=payout_upper_ratio,
        odds_cap=odds_cap,
        positive_ev_only=positive_ev_only,
        recommendation_mode=recommendation_mode,
    )
    candidates = annotated[annotated["is_recommendation_candidate"]].copy()

    selected_records = allocate_full_budget(
        candidates,
        budget=int(budget),
        unit=int(unit),
        max_bets=int(max_bets),
        target_payout=float(target_payout),
        recommendation_mode=recommendation_mode,
        payout_upper=float(payout_upper),
    )
    result = summarize_bet_set(selected_records)
    budget_remaining = int(budget) - int(result["total_stake"])
    selected_df = pd.DataFrame(result["selected_bets"])
    if not selected_df.empty:
        selected_df = selected_df.sort_values("score", ascending=False).reset_index(drop=True)

    st.header("3. 推奨買い目一覧")
    if selected_df.empty:
        st.warning(
            "現在の条件では、予算内かつ目標払戻範囲内で購入できる買い目がありません。"
            "目標払戻許容上限を広げる、推定三連複オッズ上限を広げる、最大買い目数を減らす、"
            "または予算を増やしてください。"
        )
        for reason in recommendation_empty_reasons(annotated, positive_ev_only=positive_ev_only):
            st.info(reason)
    else:
        display = selected_df[
            [
                "bet",
                "odds",
                "probability_percent",
                "theoretical_stake",
                "stake",
                "payout",
                "payout_diff",
                "expected_value",
                "score",
                "memo",
            ]
        ].rename(
            columns={
                "bet": "買い目",
                "odds": "推定三連複オッズ",
                "probability_percent": "推定的中確率（%）",
                "theoretical_stake": "理論購入額",
                "stake": "実際の推奨購入額",
                "payout": "モデル上の推定払戻額",
                "payout_diff": "目標払戻との差",
                "expected_value": "買い目期待値",
                "score": "スコア",
                "memo": "メモ",
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "推定三連複オッズ": st.column_config.NumberColumn(format="%.1f"),
                "推定的中確率（%）": st.column_config.NumberColumn(format="%.4f"),
                "理論購入額": st.column_config.NumberColumn(format="%.1f"),
                "実際の推奨購入額": st.column_config.NumberColumn(format="%d"),
                "モデル上の推定払戻額": st.column_config.NumberColumn(format="%.0f"),
                "目標払戻との差": st.column_config.NumberColumn(format="%.0f"),
                "買い目期待値": st.column_config.NumberColumn(format="%.0f"),
                "スコア": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.download_button(
            "推奨買い目結果をCSVとしてダウンロード",
            data=to_csv_bytes(selected_df),
            file_name="trifecta_recommendation_from_win_odds.csv",
            mime="text/csv",
        )

    st.header("4. セット全体のまとめ")
    col1, col2, col3 = st.columns(3)
    col1.metric("合計購入額", format_yen(result["total_stake"]))
    col2.metric("予算残り", format_yen(budget_remaining))
    col3.metric("目標払戻額", format_yen(target_payout))

    col4, col5, col6 = st.columns(3)
    col4.metric("許容推定払戻範囲", f"{format_yen(payout_lower)}〜{format_yen(payout_upper)}")
    col5.metric("推定的中確率", f"{result['hit_probability']:.4f}%")
    col6.metric("モデル上の期待払戻額", format_yen(result["expected_return"]))

    col7, col8, col9 = st.columns(3)
    col7.metric("セット期待値", format_yen(result["expected_value"]))
    col8.metric("最小推定払戻額", format_yen(result["min_payout"]))
    col9.metric("最大推定払戻額", format_yen(result["max_payout"]))

    col10, col11 = st.columns(2)
    col10.metric("平均推定払戻額", format_yen(result["average_payout"]))
    col11.metric("推奨モード", recommendation_mode)

    if not selected_df.empty:
        if budget_remaining > 0:
            st.warning(
                f"予算が{format_yen(budget_remaining)}残っています。"
                "モデル上の推定払戻額が許容上限を超えない範囲で、上位の2倍程度候補を追加・上乗せしましたが、"
                "これ以上使うと2倍程度から外れるため残しています。"
                "予算を必ず使い切りたい場合は、最大買い目数を増やす、目標払戻許容上限を広げる、"
                "推定三連複オッズ上限を広げる、または期待値プラス条件をオフにしてください。"
            )
        if result["hit_probability"] < 0.1:
            st.error(
                "推定的中確率が0.1%未満です。これは超穴狙いに近い買い方です。"
                "2倍回収を安定して狙う目的から外れている可能性があります。"
            )
        elif result["hit_probability"] < 1.0:
            st.warning(
                "推定的中確率が1%未満です。期待値は高く見えても、かなり当たりにくい買い方になっている可能性があります。"
            )

        if result["min_payout"] < payout_lower - 1e-9 or result["max_payout"] > payout_upper + 1e-9:
            st.error(
                "内部警告: 推奨買い目の最小推定払戻額または最大推定払戻額が許容範囲から外れています。"
                "これはバグの可能性があります。"
            )

    if result["hit_probability"] > 100:
        st.warning(
            f"選択セットの推定的中確率が100%を超えています（合計: {result['hit_probability']:.2f}%）。"
        )

    st.header("5. 全三連複候補ランキング")
    selected_bets = set(selected_df["bet"].tolist()) if not selected_df.empty else set()
    ranking = annotated.copy()
    ranking["selected"] = ranking["bet"].isin(selected_bets)
    # 推定三連複オッズの小さい順（人気順）に並べる。
    ranking = ranking.sort_values("odds", ascending=True)

    ranking_display = ranking[
        [
            "bet",
            "odds",
            "probability_percent",
            "theoretical_stake",
            "stake",
            "payout",
            "payout_diff",
            "expected_value",
            "score",
            "within_target_payout_range",
            "odds_cap_exceeded",
            "selected",
        ]
    ].rename(
        columns={
            "bet": "買い目",
            "odds": "推定三連複オッズ",
            "probability_percent": "推定的中確率（%）",
            "theoretical_stake": "理論購入額",
            "stake": "必要購入額",
            "payout": "モデル上の推定払戻額",
            "payout_diff": "目標払戻との差",
            "expected_value": "買い目期待値",
            "score": "スコア",
            "within_target_payout_range": "目標払戻範囲内か",
            "odds_cap_exceeded": "オッズ上限超過",
            "selected": "推奨に選択されたか",
        }
    )
    for col in ["目標払戻範囲内か", "オッズ上限超過", "推奨に選択されたか"]:
        ranking_display[col] = ranking_display[col].map(bool_to_japanese)
    st.dataframe(
        ranking_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "推定三連複オッズ": st.column_config.NumberColumn(format="%.1f"),
            "推定的中確率（%）": st.column_config.NumberColumn(format="%.4f"),
            "理論購入額": st.column_config.NumberColumn(format="%.1f"),
            "必要購入額": st.column_config.NumberColumn(format="%d"),
            "モデル上の推定払戻額": st.column_config.NumberColumn(format="%.0f"),
            "目標払戻との差": st.column_config.NumberColumn(format="%.0f"),
            "買い目期待値": st.column_config.NumberColumn(format="%.0f"),
            "スコア": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.download_button(
        "全三連複候補をCSVとしてダウンロード",
        # CSVは推定三連複オッズの昇順で出力する。
        data=to_csv_bytes(ranking.sort_values("odds", ascending=True)),
        file_name="estimated_all_trifecta_candidates.csv",
        mime="text/csv",
    )

    st.header("6. 注意事項")
    st.warning(
        "重要: この推定は単勝オッズだけを使う簡易モデルです。"
        "馬場、展開、脚質、距離適性、枠順、実際の三連複投票分布は反映していません。"
    )
    st.warning(MODEL_NOTICE)
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
