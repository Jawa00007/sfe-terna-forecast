from sfe.curate.normalize import parse_number, parse_sign, sign_from_number


def test_parse_number_decimal_comma():
    assert parse_number("118,50") == 118.5
    assert parse_number("1.234,56") == 1234.56  # it-style thousands + decimal comma
    assert parse_number("1,234.56") == 1234.56  # en-style
    assert parse_number("2 500,00") == 2500.0   # space thousands
    assert parse_number("-3,25") == -3.25


def test_parse_number_nullish_and_garbage():
    for junk in (None, "", "-", "n/d", "null", "abc", "  "):
        assert parse_number(junk) is None
    assert parse_number(float("nan")) is None


def test_parse_number_passthrough_numeric():
    assert parse_number(5) == 5.0
    assert parse_number(5.5) == 5.5


def test_parse_sign_text_and_numeric():
    assert parse_sign("+") == 1
    assert parse_sign("-") == -1
    assert parse_sign("short") == -1
    assert parse_sign("A CREDITO") == 1
    assert parse_sign("0") == 0
    assert parse_sign("12,3") == 1
    assert parse_sign("-0,4") == -1
    assert parse_sign(None) is None
    assert parse_sign("banana") is None


def test_sign_from_number():
    assert sign_from_number(10.0) == 1
    assert sign_from_number(-0.1) == -1
    assert sign_from_number(0.0) == 0
    assert sign_from_number(None) is None
