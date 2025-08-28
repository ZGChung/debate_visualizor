from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import jieba

SPEAKER_PATTERN = re.compile(
    r"^\s*(?:\[(?P<speaker_b>[^\]]+)\]|(?P<speaker_a>[^:：\[]+))\s*[:：]\s*(?P<text>.+)$"
)

STOPWORDS_EN = set(
    """
    the a an and or but if then else when where while of to in on at by for with as is are was were be being been do does did have has had from that this these those not no yes it its it's we you they he she i me my our your their them his her who whom which what why how will would can could should shall may might must also just than so such more most other into about over after before under again further once here there all any both each few own same too very s t don should now");,.?!--…“”""''()%[]{}0123456789
    """.split()
)
# Basic Chinese stopwords; for prototype purposes only
STOPWORDS_ZH = set(
    """
    的 了 和 与 及 或 而 但 就 是 在 有 没 没有 不 也 还 又 被 很 都 并 更 最 各 每 该 这 那 之 其 我 你 他 她 它 我们 你们 他们 她们 它们 吗 呢 啊 呀 吧 嘛 的话 这个 那个 一些 以及 因为 所以 如果 那么 然后 而且 但是 因此
    """.split()
)

WORD_RE_EN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
WORD_RE_ZH = re.compile(
    r"[\u4e00-\u9fff]+"
)  # Ensure full tokenization of Chinese characters

# 矛盾关系关键词（中英文）
CONTRADICTION_KEYWORDS_ZH = {
    "传统": "革新",
    "保守": "激进",
    "个人": "集体",
    "自由": "约束",
    "平等": "等级",
    "竞争": "合作",
    "效率": "公平",
    "理性": "感性",
    "物质": "精神",
    "现实": "理想",
    "统一": "多样",
    "集中": "分散",
    "君子": "小人",
    "兼爱": "仁爱",
    "非攻": "正义",
    "尚贤": "世袭",
    "科学": "迷信",  # Added more keywords
    "民主": "专制",
    "开放": "封闭",
    "进步": "落后",
}

CONTRADICTION_KEYWORDS_EN = {
    "freedom": "control",
    "individual": "collective",
    "tradition": "innovation",
    "conservative": "progressive",
    "equality": "hierarchy",
    "competition": "cooperation",
    "efficiency": "fairness",
    "rational": "emotional",
    "material": "spiritual",
    "reality": "ideal",
    "unity": "diversity",
    "centralized": "decentralized",
    "liberty": "order",
    "democracy": "authority",
    "rights": "duties",
    "science": "superstition",  # Added more keywords
    "open": "closed",
    "progress": "backward",
}


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
            utterances.append(
                Utterance(current_speaker.strip(), " ".join(buffer).strip())
            )
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
        return [
            t for t in tokens if t not in STOPWORDS_ZH and not re.fullmatch(r"\W+", t)
        ]
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


def sliding_windows(
    utterances: List[Utterance], size: int = 5, step: int = 1
) -> List[Tuple[int, List[Utterance]]]:
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


def extract_contradiction_axes(
    utterances: List[Utterance],
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """自动提取两对主要矛盾关系作为X轴和Y轴
    提取的矛盾关系用于在矛盾平面上定位文本。X轴和Y轴的值范围为-1到+1，
    其中负值表示偏向矛盾点A，正值表示偏向矛盾点B。"""
    if not utterances:
        return ("矛盾点A", "矛盾点B"), ("矛盾点C", "矛盾点D")

    lang = detect_language("\n".join(u.text for u in utterances))
    keywords = CONTRADICTION_KEYWORDS_ZH if lang == "zh" else CONTRADICTION_KEYWORDS_EN

    # 统计所有关键词的出现频率
    all_text = " ".join(u.text for u in utterances)
    tokens = tokenize(all_text, lang)
    token_freq = Counter(tokens)

    # 找到出现频率最高的两对矛盾关键词
    keyword_pairs = []
    for axis1, axis2 in keywords.items():
        score = token_freq.get(axis1, 0) + token_freq.get(axis2, 0)
        if score >= 1:  # 至少出现一次
            keyword_pairs.append((axis1, axis2, score))

    # 按频率排序，选择前两对
    keyword_pairs.sort(key=lambda x: x[2], reverse=True)

    if len(keyword_pairs) >= 2:
        x_axis = keyword_pairs[0][:2]  # (axis1, axis2)
        y_axis = keyword_pairs[1][:2]  # (axis3, axis4)
    elif len(keyword_pairs) == 1:
        x_axis = keyword_pairs[0][:2]
        # 如果没有第二对，使用默认值
        if lang == "zh":
            y_axis = ("传统观念", "革新思想")
        else:
            y_axis = ("Traditional Values", "Progressive Ideas")
    else:
        # 如果没有找到明显的矛盾关键词，使用默认值
        if lang == "zh":
            x_axis = ("传统观念", "革新思想")
            y_axis = ("个人主义", "集体主义")
        else:
            x_axis = ("Traditional Values", "Progressive Ideas")
            y_axis = ("Individualism", "Collectivism")

    return x_axis, y_axis


def calculate_contradiction_position(
    text: str, x_axis: Tuple[str, str], y_axis: Tuple[str, str], lang: str
) -> Tuple[float, float]:
    """计算文本在矛盾关系平面中的位置，使用-1到+1的坐标系统
    X轴和Y轴的负值表示偏向矛盾点A，正值表示偏向矛盾点B。"""
    tokens = tokenize(text, lang)
    token_freq = Counter(tokens)

    # Log the token frequencies for debugging
    print(f"Token frequencies: {token_freq}")

    # X轴矛盾关系：负值偏向x_axis[0]，正值偏向x_axis[1]
    x_neg_score = token_freq.get(x_axis[0], 0)  # 矛盾点A
    x_pos_score = token_freq.get(x_axis[1], 0)  # 矛盾点B

    # Log the scores for X axis
    print(f"X axis scores: {x_axis[0]}={x_neg_score}, {x_axis[1]}={x_pos_score}")

    # Y轴矛盾关系：负值偏向y_axis[0]，正值偏向y_axis[1]
    y_neg_score = token_freq.get(y_axis[0], 0)  # 矛盾点C
    y_pos_score = token_freq.get(y_axis[1], 0)  # 矛盾点D

    # Log the scores for Y axis
    print(f"Y axis scores: {y_axis[0]}={y_neg_score}, {y_axis[1]}={y_pos_score}")

    # 计算X坐标：-1到+1
    if x_neg_score == 0 and x_pos_score == 0:
        x = 0.0 + (random.random() - 0.5)  # Add small random offset
    else:
        total_x = x_neg_score + x_pos_score
        x = (x_pos_score - x_neg_score) / total_x  # 范围：-1到+1

    # 计算Y坐标：-1到+1
    if y_neg_score == 0 and y_pos_score == 0:
        y = 0.0 + (random.random() - 0.5)  # Add small random offset
    else:
        total_y = y_neg_score + y_pos_score
        y = (y_pos_score - y_neg_score) / total_y  # 范围：-1到+1

    return x, y


def contradiction_analysis_series(utterances: List[Utterance]) -> List[Dict[str, any]]:
    """分析每轮对话在矛盾关系平面中的位置
    使用提取的矛盾关系作为坐标轴，计算每个文本在平面中的位置。"""
    if not utterances:
        return []

    lang = detect_language("\n".join(u.text for u in utterances))
    x_axis, y_axis = extract_contradiction_axes(utterances)

    series = []
    for i, utterance in enumerate(utterances):
        x, y = calculate_contradiction_position(utterance.text, x_axis, y_axis, lang)
        series.append(
            {
                "index": i,
                "speaker": utterance.speaker,
                "x": x,
                "y": y,
                "x_axis": x_axis,
                "y_axis": y_axis,
                "text": (
                    utterance.text[:50] + "..."
                    if len(utterance.text) > 50
                    else utterance.text
                ),
            }
        )

    return series
