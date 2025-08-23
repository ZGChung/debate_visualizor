from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jieba

SPEAKER_PATTERN = re.compile(r"^\s*(?:\[(?P<speaker_b>[^\]]+)\]|(?P<speaker_a>[^:：\[]+))\s*[:：]\s*(?P<text>.+)$")

STOPWORDS_EN = set(
    """
    the a an and or but if then else when where while of to in on at by for with as is are was were be being been do does did have has had from that this these those not no yes it its it's we you they he she i me my our your their them his her who whom which what why how will would can could should shall may might must also just than so such more most other into about over after before under again further once here there all any both each few own same too very s t don should now");,.?!--…“”""''()%[]{}0123456789
    """
    .split()
)
# Basic Chinese stopwords; for prototype purposes only
STOPWORDS_ZH = set(
    """
    的 了 和 与 及 或 而 但 就 是 在 有 没 没有 不 也 还 又 被 很 都 并 更 最 各 每 该 这 那 之 其 我 你 他 她 它 我们 你们 他们 她们 它们 吗 呢 啊 呀 吧 嘛 的话 这个 那个 一些 以及 因为 所以 如果 那么 然后 而且 但是 因此
    """
    .split()
)

WORD_RE_EN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WORD_RE_ZH = re.compile(r"[\u4e00-\u9fff]")


def detect_language(text: str) -> str:
    if re.search(WORD_RE_ZH, text):
        return "zh"
    return "en"


@dataclass
class Utterance:
    speaker: str
    text: str


def parse_transcript(raw: str) -> List[Utterance]:
    """Parse transcript lines in format like:
    Alice: Opening statement
    [Bob]: Rebuttal here
    Lines without a leading speaker label are attached to the previous speaker.
    """
    utterances: List[Utterance] = []
    current_speaker = "Unknown"
    buffer: List[str] = []

    def flush():
        nonlocal buffer, current_speaker
        if buffer:
            utterances.append(Utterance(current_speaker.strip(), " ".join(buffer).strip()))
            buffer = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_PATTERN.match(line)
        if m:
            flush()
            speaker = m.group("speaker_a") or m.group("speaker_b") or "Unknown"
            current_speaker = speaker.strip()
            buffer.append(m.group("text"))
        else:
            buffer.append(line)
    flush()
    return utterances


def tokenize(text: str, lang: str) -> List[str]:
    if lang == "zh":
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        return [t for t in tokens if t not in STOPWORDS_ZH and not re.fullmatch(r"\W+", t)]
    # English
    tokens = [t.lower() for t in re.findall(WORD_RE_EN, text)]
    return [t for t in tokens if t not in STOPWORDS_EN]


def word_frequencies(utterances: List[Utterance]) -> Dict[str, int]:
    if not utterances:
        return {}
    # detect lang by concatenating text
    big_text = "\n".join(u.text for u in utterances)
    lang = detect_language(big_text)
    counter: Counter[str] = Counter()
    for u in utterances:
        counter.update(tokenize(u.text, lang))
    return dict(counter)


def top_speakers(utterances: List[Utterance], k: int = 2) -> List[str]:
    counts: Counter[str] = Counter(u.speaker for u in utterances)
    return [s for s, _ in counts.most_common(k)]


def sliding_windows(utterances: List[Utterance], size: int = 5, step: int = 1) -> List[Tuple[int, List[Utterance]]]:
    res = []
    for i in range(0, max(0, len(utterances) - size + 1), step):
        res.append((i, utterances[i : i + size]))
    if not res and utterances:
        res.append((0, utterances))
    return res


def tug_of_war_series(utterances: List[Utterance]) -> List[Dict[str, float]]:
    """Compute a time series showing semantic pull between two main speakers.
    Approach: compute log odds ratio of each window belonging to speaker A vs B using unigram Naive Bayes-like counts.
    """
    if not utterances:
        return []

    speakers = top_speakers(utterances, 2)
    if len(speakers) < 2:
        return [{"index": 0, "score": 0.0}]
    sA, sB = speakers[0], speakers[1]

    # Build per-speaker token counts
    data_by_s = {s: [] for s in speakers}
    for u in utterances:
        data_by_s[u.speaker].append(u.text)

    lang = detect_language("\n".join(u.text for u in utterances))

    def counts_for(texts: List[str]) -> Counter[str]:
        c: Counter[str] = Counter()
        for t in texts:
            c.update(tokenize(t, lang))
        return c

    cA = counts_for(data_by_s.get(sA, []))
    cB = counts_for(data_by_s.get(sB, []))

    vocab = set(cA) | set(cB)
    alpha = 0.5
    totalA = sum(cA.values()) + alpha * len(vocab)
    totalB = sum(cB.values()) + alpha * len(vocab)

    def log_prob_speaker(tokens: List[str], c: Counter[str], total: float) -> float:
        import math

        s = 0.0
        for tok in tokens:
            s += math.log((c.get(tok, 0) + alpha) / total)
        return s

    series: List[Dict[str, float]] = []
    for idx, window in sliding_windows(utterances, size=5, step=1):
        tokens = []
        for u in window:
            tokens.extend(tokenize(u.text, lang))
        lpA = log_prob_speaker(tokens, cA, totalA)
        lpB = log_prob_speaker(tokens, cB, totalB)
        score = float(lpA - lpB)  # positive => A pull; negative => B pull
        series.append({"index": idx, "score": score, "speakerA": sA, "speakerB": sB})

    return series
