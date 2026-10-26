"""
Post-Processing and Text Correction for DeepOCR.

Operations (in order):
  1. Unicode normalisation (NFKC) and typographic punctuation -> ASCII.
  2. Context-aware character confusion fixes — digits inside words become
     letters ("0ffice" -> "Office", "he1lo" -> "hello") and letters inside
     numbers become digits ("2O24" -> "2024", "l5" -> "15").  Pure numbers,
     codes and amounts are never rewritten.
  3. Whitespace normalisation.
  4. Word-level spell checking with pyspellchecker (optional), restricted to
     lowercase/capitalised alphabetic words of 3+ letters, preserving case and
     punctuation, skipping ALL-CAPS acronyms and likely proper nouns, and
     only accepting candidates at edit distance 1 (configurable).
  5. Confidence-based filtering of low-confidence characters (optional).

``correct_with_report`` returns the corrected text *and* a list of the
individual changes so a reviewer can see (and undo) every automatic edit.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

_TYPOGRAPHY = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}
# Visually confusable characters, by direction of the fix.
_DIGIT_TO_LETTER = {"0": "o", "1": "l", "5": "s", "8": "B"}
_LETTER_TO_DIGIT = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2"}
_TOKEN = re.compile(r"\S+")
_CORE = re.compile(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$")
# "name.surnameOcollege.edu" -> "name.surname@college.edu" (@ misread as O/0/a)
_EMAIL_AT = re.compile(
    r"^([A-Za-z0-9._%+-]*[a-z0-9._%+-])[O0Q@]([a-z0-9-]+\.(?:com|edu|org|net|in|ac\.in|co\.in|gov|io|ai)\.?)$"
)


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    kind: str  # "confusion" | "spelling"
    position: int  # character offset in the corrected text

    def to_dict(self) -> dict:
        return {"original": self.original, "corrected": self.corrected,
                "kind": self.kind, "position": self.position}


@lru_cache(maxsize=1)
def _spellchecker(distance: int):
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return None
    return SpellChecker(distance=distance)


def _match_case(template: str, word: str) -> str:
    if template.isupper():
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


class TextCorrector:
    """Apply conservative post-processing corrections to raw OCR text.

    Config keys (defaults):
        apply_char_subs   (True)  Context-aware digit/letter confusion fixes.
        apply_spell_check (True)  pyspellchecker word correction.
        apply_normalise   (False) Lowercase + strip special characters.
        min_confidence    (0.0)   Characters below this confidence become spaces.
        spell_distance    (1)     Max edit distance for spelling candidates.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.apply_char_subs = cfg.get("apply_char_subs", True)
        self.apply_spell_check = cfg.get("apply_spell_check", True)
        self.apply_normalise = cfg.get("apply_normalise", False)
        self.min_confidence = cfg.get("min_confidence", 0.0)
        self.spell_distance = cfg.get("spell_distance", 1)
        self._spellchecker = _spellchecker(self.spell_distance) if self.apply_spell_check else None

    @property
    def spellcheck_available(self) -> bool:
        return self._spellchecker is not None

    def correct(self, text: str, confidences: list[float] | None = None) -> str:
        """Return corrected text (see module docstring for the steps)."""
        return self.correct_with_report(text, confidences)[0]

    def correct_with_report(
        self, text: str, confidences: list[float] | None = None
    ) -> tuple[str, list[Correction]]:
        if not text:
            return text, []
        if confidences and self.min_confidence > 0:
            text = self._confidence_filter(text, confidences, self.min_confidence)
        text = unicodedata.normalize("NFKC", text)
        for wrong, right in _TYPOGRAPHY.items():
            text = text.replace(wrong, right)
        text = self._normalise_whitespace(text)

        corrections: list[Correction] = []
        output: list[str] = []
        cursor = 0
        for match in _TOKEN.finditer(text):
            output.append(text[cursor:match.start()])
            token = match.group(0)
            fixed, kind = self._fix_token(token)
            position = sum(len(part) for part in output)
            if fixed != token:
                corrections.append(Correction(token, fixed, kind, position))
            output.append(fixed)
            cursor = match.end()
        output.append(text[cursor:])
        result = "".join(output)
        if self.apply_normalise:
            result = self._normalise_text(result)
        return result.strip(), corrections

    # ------------------------------------------------------------------
    # Token-level fixes
    # ------------------------------------------------------------------

    def _fix_token(self, token: str) -> tuple[str, str]:
        if self.apply_char_subs:
            email = _EMAIL_AT.match(token)
            if email and "@" not in token:
                return f"{email.group(1)}@{email.group(2)}", "confusion"
        prefix, core, suffix = _CORE.match(token).groups()
        if not core:
            return token, ""
        kind = ""
        if self.apply_char_subs:
            fixed = self._fix_confusions(core)
            if fixed != core:
                core, kind = fixed, "confusion"
        if self._spellchecker is not None:
            fixed = self._spell(core)
            if fixed != core:
                core, kind = fixed, "spelling"
        return f"{prefix}{core}{suffix}", kind

    @staticmethod
    def _fix_confusions(core: str) -> str:
        letters = sum(c.isalpha() for c in core)
        digits = sum(c.isdigit() for c in core)
        if not letters or not digits or not core.isalnum():
            return core
        if letters >= 3 and digits == 1:
            # A single stray digit inside a word: "he1lo", "0ffice"
            candidate = "".join(_DIGIT_TO_LETTER.get(c, c) if c.isdigit() else c for c in core)
            if not candidate.isalpha():
                return core
            if core.isupper() or (core[0].isdigit() and core[1:].isupper()):
                return candidate.upper()
            if core[0].isdigit():
                return candidate[0].upper() + candidate[1:]
            return candidate
        if digits >= 2 and letters == 1:
            # A single confusable letter inside a number: "2O24", "l5"
            chars = [(_LETTER_TO_DIGIT.get(c, c) if c.isalpha() else c) for c in core]
            candidate = "".join(chars)
            return candidate if candidate.isdigit() else core
        return core

    def _spell(self, core: str) -> str:
        if not core.isalpha() or len(core) < 3 or core.isupper():
            return core
        if core[0].isupper() and core[1:].islower() is False:
            return core  # mixed case such as "McDonald" / "iPhone"
        lower = core.lower()
        checker = self._spellchecker
        if lower in checker:
            return core
        candidates = {c for c in (checker.candidates(lower) or set()) if c != lower}
        if not candidates:
            return core
        best = max(candidates, key=checker.word_usage_frequency)
        # Capitalised unknown words are often names: only fix them towards common words.
        if core[0].isupper() and checker.word_usage_frequency(best) < 1e-6:
            return core
        return _match_case(core, best)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_filter(text: str, confidences: list[float], threshold: float) -> str:
        """Replace low-confidence characters with a space."""
        chars = list(text)
        for i, conf in enumerate(confidences):
            if i < len(chars) and conf < threshold:
                chars[i] = " "
        return "".join(chars)

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """Collapse multiple spaces/tabs and long blank-line runs."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Lowercase and remove non-alphanumeric non-space characters."""
        text = text.lower()
        return re.sub(r"[^a-z0-9 .,!?\n]", "", text)
