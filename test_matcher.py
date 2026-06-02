"""Offline tests for the pure logic in matcher.py (no network needed).

Run with pytest, or directly: python test_matcher.py
"""

import matcher


def test_parse_fit_score():
    assert matcher.parse_fit_score("87") == 87
    assert matcher.parse_fit_score("Score: 42 (high)") == 42
    assert matcher.parse_fit_score("150") == 100  # clamped to 100
    assert matcher.parse_fit_score("") == 0
    assert matcher.parse_fit_score("no number here") == 0


def test_select_threshold():
    assert matcher.select_threshold([80] * 20) == 80          # enough strong matches
    assert matcher.select_threshold([80] * 19 + [70]) == 75   # < 20 at 80, but some >= 75
    assert matcher.select_threshold([76]) == 75
    assert matcher.select_threshold([10, 20, 74]) is None     # nothing reaches 75


def test_parse_refined_scores():
    raw = "Journal 1: 95\nJournal 2: 88\nJournal 3: 70"
    assert matcher.parse_refined_scores(raw, 3) == {0: 95, 1: 88, 2: 70}
    assert matcher.parse_refined_scores("Journal 5: 90", 3) == {}   # out of range ignored
    assert matcher.parse_refined_scores("", 3) == {}


def test_refine_scores_keeps_original_on_mismatch():
    shortlist = [{"Name": "A", "Fit": 80}, {"Name": "B", "Fit": 82}]
    # No api_key/network is used because we monkeypatch the network call.
    matcher._post = lambda *a, **k: ("Journal 1: 91", None)  # only 1 of 2 -> mismatch
    assert matcher.refine_scores("summary", shortlist, "key") == [80, 82]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
