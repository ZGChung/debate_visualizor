import math
from backend.utils import parse_transcript, detect_language, tokenize, word_frequencies, tug_of_war_series, Utterance


def test_parse_transcript_basic():
    raw = """
    Alice: Hello there.
    Bob: Hi! Nice to meet you.
    Alice: Likewise.
    """.strip()
    utts = parse_transcript(raw)
    assert len(utts) == 3
    assert utts[0].speaker == 'Alice'
    assert 'Hello' in utts[0].text


def test_detect_language_en_zh():
    assert detect_language('Hello world') == 'en'
    assert detect_language('你好 世界') == 'zh'


def test_tokenize_en():
    toks = tokenize("This is an apple, and it's great!", 'en')
    assert 'apple' in toks
    assert 'is' not in toks


def test_word_frequencies_count():
    utts = [Utterance('A', 'apple banana'), Utterance('B', 'banana apple apple')]
    freqs = word_frequencies(utts)
    assert freqs['apple'] == 3
    assert freqs['banana'] == 2


def test_tug_of_war_series():
    raw = """
    Alice: economy growth tax
    Bob: policy tax regulation
    Alice: jobs growth investment
    Bob: welfare policy healthcare
    Alice: growth market export
    """.strip()
    utts = parse_transcript(raw)
    series = tug_of_war_series(utts)
    assert isinstance(series, list)
    assert len(series) >= 1
    assert 'score' in series[0]
