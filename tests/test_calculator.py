from src.calculator import (
    calculate_expected_value,
    calculate_payout,
    calculate_required_stake,
    ceil_to_unit,
)


def test_ceil_to_unit_rounds_up():
    assert ceil_to_unit(1111.11, 100) == 1200


def test_ceil_to_unit_exact_value():
    assert ceil_to_unit(2000, 100) == 2000


def test_required_stake_for_target_payout():
    assert calculate_required_stake(target_payout=20000, odds=10.0, unit=100) == 2000


def test_expected_value():
    payout = calculate_payout(stake=2000, odds=10.0)
    assert calculate_expected_value(probability_percent=15.0, payout=payout, stake=2000) == 1000
