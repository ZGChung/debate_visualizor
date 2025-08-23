from __future__ import annotations

from typing import Any, Dict, List

from .utils import Utterance, parse_transcript, tug_of_war_series, word_frequencies


def analyze_transcript(raw: str) -> Dict[str, Any]:
    utterances: List[Utterance] = parse_transcript(raw)
    freqs = word_frequencies(utterances)
    tug = tug_of_war_series(utterances)
    # prepare a sorted list for top-N words for word cloud
    top_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:200]
    return {
        "utterances": [u.__dict__ for u in utterances],
        "frequencies": freqs,
        "top_words": top_words,
        "tug_of_war": tug,
    }
