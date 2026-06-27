import pandas as pd

from src.optimizer import optimize_bets
from src.recommendation import (
    EV_MODE,
    HIT_RATE_MODE,
    STABLE_MODE,
    allocate_full_budget,
    annotate_candidates,
)


def _base_candidates():
    return pd.DataFrame(
        [
            {
                "bet": "close",
                "odds": 40.0,
                "probability_percent": 5.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 4000.0,
                "payout_diff": 0.0,
                "expected_value": 100.0,
                "memo": "",
            },
            {
                "bet": "upper-edge",
                "odds": 52.0,
                "probability_percent": 5.0,
                "theoretical_stake": 76.9,
                "stake": 100,
                "payout": 5200.0,
                "payout_diff": 1200.0,
                "expected_value": 160.0,
                "memo": "",
            },
            {
                "bet": "too-high-payout",
                "odds": 1000.0,
                "probability_percent": 0.2,
                "theoretical_stake": 4.0,
                "stake": 100,
                "payout": 100000.0,
                "payout_diff": 96000.0,
                "expected_value": 100.0,
                "memo": "",
            },
        ]
    )


def _annotate(df, mode=STABLE_MODE, odds_cap=200.0, positive_ev_only=False):
    return annotate_candidates(
        df,
        budget=2000,
        target_payout=4000,
        payout_lower_ratio=0.9,
        payout_upper_ratio=1.3,
        odds_cap=odds_cap,
        positive_ev_only=positive_ev_only,
        recommendation_mode=mode,
    )


def test_only_bets_within_target_payout_range_are_recommendation_candidates():
    annotated = _annotate(_base_candidates(), odds_cap=2000.0)

    selected = annotated[annotated["is_recommendation_candidate"]]["bet"].tolist()

    assert "close" in selected
    assert "upper-edge" in selected
    assert "too-high-payout" not in selected
    assert not annotated.loc[annotated["bet"] == "too-high-payout", "within_target_payout_range"].iloc[0]


def test_ultra_high_payout_bet_is_excluded_even_if_expected_value_is_positive():
    annotated = _annotate(_base_candidates(), odds_cap=2000.0, positive_ev_only=True)

    assert annotated.loc[
        annotated["bet"] == "too-high-payout", "expected_value"
    ].iloc[0] > 0
    assert not annotated.loc[
        annotated["bet"] == "too-high-payout", "is_recommendation_candidate"
    ].iloc[0]


def test_odds_cap_excludes_bets_from_recommendations():
    annotated = _annotate(_base_candidates(), odds_cap=50.0)

    assert annotated.loc[annotated["bet"] == "upper-edge", "odds_cap_exceeded"].iloc[0]
    assert not annotated.loc[
        annotated["bet"] == "upper-edge", "is_recommendation_candidate"
    ].iloc[0]


def test_stable_mode_prefers_payout_close_to_target():
    annotated = _annotate(_base_candidates(), mode=STABLE_MODE, odds_cap=200.0)
    result = optimize_bets(
        annotated[annotated["is_recommendation_candidate"]],
        budget=2000,
        max_bets=1,
        positive_ev_only=False,
        value_column="score",
    )

    assert [bet["bet"] for bet in result["selected_bets"]] == ["close"]


def test_ev_mode_prefers_higher_expected_value():
    annotated = _annotate(_base_candidates(), mode=EV_MODE, odds_cap=200.0)
    result = optimize_bets(
        annotated[annotated["is_recommendation_candidate"]],
        budget=2000,
        max_bets=1,
        positive_ev_only=False,
        value_column="score",
    )

    assert [bet["bet"] for bet in result["selected_bets"]] == ["upper-edge"]


def test_hit_rate_mode_prefers_higher_hit_probability():
    df = pd.DataFrame(
        [
            {
                "bet": "low-prob-high-ev",
                "odds": 40.0,
                "probability_percent": 2.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 4000.0,
                "payout_diff": 0.0,
                "expected_value": 500.0,
                "memo": "",
            },
            {
                "bet": "high-prob-low-ev",
                "odds": 40.0,
                "probability_percent": 10.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 4000.0,
                "payout_diff": 0.0,
                "expected_value": 0.0,
                "memo": "",
            },
        ]
    )
    annotated = _annotate(df, mode=HIT_RATE_MODE, odds_cap=200.0)
    result = optimize_bets(
        annotated[annotated["is_recommendation_candidate"]],
        budget=2000,
        max_bets=1,
        positive_ev_only=False,
        value_column="score",
    )

    assert [bet["bet"] for bet in result["selected_bets"]] == ["high-prob-low-ev"]


def test_allocate_full_budget_buys_additional_two_times_candidates_before_topping_up():
    df = pd.DataFrame(
        [
            {
                "bet": "top",
                "odds": 20.0,
                "probability_percent": 10.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 2000.0,
                "payout_diff": 0.0,
                "expected_value": 100.0,
                "score": 100.0,
                "memo": "",
            },
            {
                "bet": "second",
                "odds": 20.0,
                "probability_percent": 9.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 2000.0,
                "payout_diff": 0.0,
                "expected_value": 80.0,
                "score": 80.0,
                "memo": "",
            },
            {
                "bet": "third",
                "odds": 20.0,
                "probability_percent": 8.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 2000.0,
                "payout_diff": 0.0,
                "expected_value": 60.0,
                "score": 60.0,
                "memo": "",
            },
        ]
    )

    selected = allocate_full_budget(
        df,
        budget=300,
        unit=100,
        max_bets=3,
        target_payout=2000,
        recommendation_mode=STABLE_MODE,
        payout_upper=2600,
    )

    assert [bet["bet"] for bet in selected] == ["top", "second", "third"]
    assert sum(bet["stake"] for bet in selected) == 300


def test_allocate_full_budget_does_not_top_up_beyond_payout_upper():
    df = pd.DataFrame(
        [
            {
                "bet": "only",
                "odds": 20.0,
                "probability_percent": 10.0,
                "theoretical_stake": 100.0,
                "stake": 100,
                "payout": 2000.0,
                "payout_diff": 0.0,
                "expected_value": 100.0,
                "score": 100.0,
                "memo": "",
            }
        ]
    )

    selected = allocate_full_budget(
        df,
        budget=500,
        unit=100,
        max_bets=1,
        target_payout=2000,
        recommendation_mode=STABLE_MODE,
        payout_upper=2600,
    )

    assert len(selected) == 1
    assert selected[0]["stake"] == 100
    assert selected[0]["payout"] == 2000.0
