import math

from src.estimator import (
    build_trifecta_candidates,
    estimate_trifecta_odds,
    harville_trifecta_probabilities,
    win_odds_to_win_probabilities,
)


def test_win_odds_to_win_probabilities_sum_to_one():
    probabilities = win_odds_to_win_probabilities({1: 2.0, 2: 4.0, 3: 8.0})

    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities[1] > probabilities[2] > probabilities[3]


def test_harville_three_horses_probability_is_one():
    probabilities = win_odds_to_win_probabilities({1: 2.0, 2: 4.0, 3: 8.0})
    trifecta = harville_trifecta_probabilities(probabilities)

    assert list(trifecta.keys()) == [(1, 2, 3)]
    assert math.isclose(trifecta[(1, 2, 3)], 1.0)


def test_estimate_trifecta_odds_from_probability():
    assert estimate_trifecta_odds(0.1, takeout=0.25) == 7.5


def test_build_trifecta_candidates_generates_combinations():
    df = build_trifecta_candidates({1: 2.0, 2: 4.0, 3: 8.0, 4: 16.0})

    assert len(df) == 4
    assert set(df.columns) == {
        "bet",
        "odds",
        "probability_percent",
        "market_probability_percent",
        "memo",
    }
    assert (df["odds"] > 1.0).all()
    assert (df["probability_percent"] > 0).all()


def test_two_model_estimator_can_create_probability_market_gap():
    same = build_trifecta_candidates(
        {1: 2.0, 2: 4.0, 3: 8.0, 4: 16.0, 5: 32.0},
        belief_lambda=1.0,
        market_lambda=1.0,
        takeout=0.25,
    )
    shifted = build_trifecta_candidates(
        {1: 2.0, 2: 4.0, 3: 8.0, 4: 16.0, 5: 32.0},
        belief_lambda=0.7,
        market_lambda=1.0,
        takeout=0.25,
    )

    merged = same.merge(shifted, on="bet", suffixes=("_same", "_shifted"))
    assert (merged["probability_percent_same"] != merged["probability_percent_shifted"]).any()
    assert (merged["odds_same"] == merged["odds_shifted"]).all()
