import pytest

from clipwright.platforms import DEFAULT_MAX, DEFAULT_MIN, PLATFORMS, lengths


def test_no_platform_uses_the_old_defaults():
    assert lengths(None, None, None) == (DEFAULT_MIN, DEFAULT_MAX)
    assert lengths(None, 10, None) == (10, DEFAULT_MAX)


@pytest.mark.parametrize("name", list(PLATFORMS))
def test_presets_stay_under_the_hard_limit(name: str):
    p = PLATFORMS[name]
    assert 0 < p.min_len <= p.max_len <= p.limit
    assert lengths(name, None, None) == (p.min_len, p.max_len)


def test_explicit_values_win():
    assert lengths("tiktok", 30, 120) == (30, 120)


def test_max_over_the_limit_is_rejected():
    with pytest.raises(ValueError, match="at most 60s"):
        lengths("shorts", None, 75)


def test_min_above_the_preset_max_stretches_the_max():
    assert lengths("shorts", 59, None) == (59, 59)
    assert lengths("reels", 70, None) == (70, 70)


def test_unknown_platform():
    with pytest.raises(ValueError, match="unknown platform"):
        lengths("myspace", None, None)
