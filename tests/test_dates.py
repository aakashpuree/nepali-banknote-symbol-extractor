from nepali_banknote_scanner.dates import extract_date_candidates, to_ascii_digits


def test_converts_nepali_digits_without_changing_other_text() -> None:
    assert to_ascii_digits("मिति २०८१/०५/०६") == "मिति 2081/05/06"


def test_does_not_treat_serial_numbers_as_dates() -> None:
    assert extract_date_candidates("serial no 660302") == []


def test_auto_calendar_never_guesses_bs_or_ad() -> None:
    result = extract_date_candidates("मिति २०७९/०५/१५")[0]
    assert result.ascii_value == "2079/05/15"
    assert result.source_calendar == "unknown"
    assert result.normalized_date is None
    assert result.requires_review is True


def test_invalid_ad_date_is_rejected() -> None:
    result = extract_date_candidates("2023/02/30", "ad")[0]
    assert result.valid is False
