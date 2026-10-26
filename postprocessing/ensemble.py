"""
Multi-model word voting (ROVER: Recognizer Output Voting Error Reduction, Fiscus 1997).

Several recognisers read the same text line. Their outputs are aligned word by
word into a *word transition network*: a sequence of slots, each holding
the words the models proposed at that position, or an empty vote ("no word
here"). Each slot is decided by weighted voting, so a word proposed by several
independent models beats one proposed by a single model. Each output word
carries its *support*: the share of total model weight that voted for it.

Alignment uses a word-level edit distance whose substitution cost is the
normalised character distance between words. Near-misses ("assuredness" /
"assurances") therefore line up in the same slot instead of being treated as
an insertion plus a deletion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

EMPTY = ""  # vote for "no word in this slot"


@dataclass(frozen=True)
class Hypothesis:
    text: str
    source: str
    weight: float = 1.0


@dataclass
class VotedWord:
    text: str
    support: float                      # share of total weight that chose this word (0..1)
    sources: list[str]                  # models that proposed exactly this word
    alternatives: dict[str, float] = field(default_factory=dict)  # other candidates and their share

    def to_dict(self) -> dict:
        return {"text": self.text, "support": round(self.support, 3), "sources": self.sources,
                "alternatives": {k: round(v, 3) for k, v in self.alternatives.items()}}


@dataclass
class _Slot:
    votes: dict[str, float] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)

    def add(self, word: str, weight: float, source: str | None = None) -> None:
        self.votes[word] = self.votes.get(word, 0.0) + weight
        if source is not None:
            self.sources.setdefault(word, []).append(source)

    def best(self) -> str:
        return max(self.votes.items(), key=lambda item: item[1])[0]


@lru_cache(maxsize=200_000)
def _word_distance(a: str, b: str) -> float:
    """Normalised character edit distance between two words (0 = identical, 1 = unrelated)."""
    if a == b:
        return 0.0
    x, y = a.lower(), b.lower()
    if x == y:
        return 0.1  # case-only difference: same slot, different vote
    if not x or not y:
        return 1.0
    previous = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        current = [i]
        for j, cy in enumerate(y, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (cx != cy)))
        previous = current
    return min(1.0, previous[-1] / max(len(x), len(y)))


def _align(words: list[str], slots: list[str]) -> list[tuple[int | None, int | None]]:
    """Edit-distance alignment of ``words`` against slot representatives.

    Returns (word_index, slot_index) pairs; None marks an insertion or deletion.
    Slots whose current winner is EMPTY are cheap to skip.
    """
    n, m = len(words), len(slots)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + (0.3 if slots[j - 1] == EMPTY else 1.0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            skip = 0.3 if slots[j - 1] == EMPTY else 1.0
            sub = 1.0 if slots[j - 1] == EMPTY else _word_distance(words[i - 1], slots[j - 1]) * 1.5
            cost[i][j] = min(cost[i - 1][j - 1] + sub, cost[i - 1][j] + 1.0, cost[i][j - 1] + skip)
    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        skip = 0.3 if j > 0 and slots[j - 1] == EMPTY else 1.0
        if i > 0 and j > 0:
            sub = 1.0 if slots[j - 1] == EMPTY else _word_distance(words[i - 1], slots[j - 1]) * 1.5
            if abs(cost[i][j] - (cost[i - 1][j - 1] + sub)) < 1e-9:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if j > 0 and abs(cost[i][j] - (cost[i][j - 1] + skip)) < 1e-9:
            pairs.append((None, j - 1))
            j -= 1
        else:
            pairs.append((i - 1, None))
            i -= 1
    pairs.reverse()
    return pairs


def rover(hypotheses: list[Hypothesis]) -> tuple[str, list[VotedWord]]:
    """Combine several readings of one line. Returns (text, per-word votes)."""
    hyps = [h for h in hypotheses if h.weight > 0]
    if not hyps:
        return "", []
    hyps.sort(key=lambda h: h.weight, reverse=True)
    total = sum(h.weight for h in hyps)

    pivot = hyps[0]
    slots: list[_Slot] = []
    for word in pivot.text.split():
        slot = _Slot()
        slot.add(word, pivot.weight, pivot.source)
        slots.append(slot)
    seen = pivot.weight

    for hyp in hyps[1:]:
        words = hyp.text.split()
        pairs = _align(words, [slot.best() for slot in slots])
        merged: list[_Slot] = []
        for word_index, slot_index in pairs:
            if slot_index is None:  # word not present in any slot: open a new slot
                slot = _Slot()
                slot.add(EMPTY, seen)
                slot.add(words[word_index], hyp.weight, hyp.source)
                merged.append(slot)
                continue
            slot = slots[slot_index]
            if word_index is None:
                slot.add(EMPTY, hyp.weight)
            else:
                slot.add(words[word_index], hyp.weight, hyp.source)
            merged.append(slot)
        slots = merged
        seen += hyp.weight

    voted: list[VotedWord] = []
    for slot in slots:
        # Highest total weight wins; ties go to the higher-weighted model's word.
        ranked = sorted(slot.votes.items(), key=lambda item: (item[1], _first_weight(item[0], slot, hyps)), reverse=True)
        word, weight = ranked[0]
        if word == EMPTY:
            continue
        voted.append(VotedWord(
            text=word,
            support=weight / total,
            sources=slot.sources.get(word, []),
            alternatives={w: v / total for w, v in ranked[1:] if w != EMPTY},
        ))
    return " ".join(v.text for v in voted), voted


def _first_weight(word: str, slot: _Slot, hyps: list[Hypothesis]) -> float:
    sources = slot.sources.get(word, [])
    return max((h.weight for h in hyps if h.source in sources), default=0.0)
