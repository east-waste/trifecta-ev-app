from src.ocr import image_bytes_to_data_url, normalize_ocr_records


def test_image_bytes_to_data_url():
    data_url = image_bytes_to_data_url(b"abc", filename="odds.png")

    assert data_url.startswith("data:image/png;base64,")
    assert data_url.endswith("YWJj")


def test_normalize_ocr_records():
    df = normalize_ocr_records(
        [
            {"horse_number": "2", "horse_name": "サンプルホース", "win_odds": "18.0倍"},
            {"馬番": 1, "馬名": "テストホース", "単勝オッズ": "3.1"},
            {"horse_number": "", "win_odds": ""},
        ]
    )

    assert df["horse_number"].tolist() == [1, 2]
    assert df["horse_name"].tolist() == ["テストホース", "サンプルホース"]
    assert df["win_odds"].tolist() == [3.1, 18.0]
