from src.optimizer import optimize_bets


def test_selects_max_expected_value_within_budget():
    candidates = [
        {"bet": "A", "stake": 1000, "payout": 20000, "probability_percent": 5.0, "expected_value": 0},
        {"bet": "B", "stake": 1000, "payout": 20000, "probability_percent": 10.0, "expected_value": 1000},
        {"bet": "C", "stake": 2000, "payout": 22000, "probability_percent": 12.0, "expected_value": 640},
    ]

    result = optimize_bets(candidates, budget=1000, max_bets=1, positive_ev_only=True)

    assert [bet["bet"] for bet in result["selected_bets"]] == ["B"]
    assert result["expected_value"] == 1000


def test_excludes_bets_over_budget():
    candidates = [
        {"bet": "A", "stake": 12000, "payout": 24000, "probability_percent": 80.0, "expected_value": 7200},
        {"bet": "B", "stake": 1000, "payout": 20000, "probability_percent": 10.0, "expected_value": 1000},
    ]

    result = optimize_bets(candidates, budget=10000, max_bets=5, positive_ev_only=True)

    assert [bet["bet"] for bet in result["selected_bets"]] == ["B"]
    assert result["total_stake"] == 1000


def test_respects_max_bets():
    candidates = [
        {"bet": "A", "stake": 1000, "payout": 20000, "probability_percent": 10.0, "expected_value": 1000},
        {"bet": "B", "stake": 1000, "payout": 20000, "probability_percent": 9.0, "expected_value": 800},
        {"bet": "C", "stake": 1000, "payout": 20000, "probability_percent": 8.0, "expected_value": 600},
    ]

    result = optimize_bets(candidates, budget=10000, max_bets=2, positive_ev_only=True)

    assert len(result["selected_bets"]) <= 2
    assert {bet["bet"] for bet in result["selected_bets"]} == {"A", "B"}


def test_positive_ev_only_excludes_negative_ev():
    candidates = [
        {"bet": "A", "stake": 1000, "payout": 20000, "probability_percent": 1.0, "expected_value": -800},
        {"bet": "B", "stake": 1000, "payout": 20000, "probability_percent": 10.0, "expected_value": 1000},
    ]

    result = optimize_bets(candidates, budget=10000, max_bets=5, positive_ev_only=True)

    assert [bet["bet"] for bet in result["selected_bets"]] == ["B"]
    assert all(bet["expected_value"] > 0 for bet in result["selected_bets"])
