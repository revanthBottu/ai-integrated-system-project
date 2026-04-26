# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Text-based emoji patterns mapped to a sentiment signal token.
TEXT_EMOJI_MAP: Dict[str, str] = {
    ":)": "__emoji_positive__",
    ":-)": "__emoji_positive__",
    ":D": "__emoji_positive__",
    ":-D": "__emoji_positive__",
    ":P": "__emoji_positive__",
    ":(": "__emoji_negative__",
    ":-(": "__emoji_negative__",
    ":'(": "__emoji_negative__",
    ":/": "__emoji_negative__",
}

# Unicode emoji characters mapped to a sentiment signal token.
UNICODE_EMOJI_MAP: Dict[str, str] = {
    "😊": "__emoji_positive__",
    "😄": "__emoji_positive__",
    "😁": "__emoji_positive__",
    "😂": "__emoji_positive__",
    "🎉": "__emoji_positive__",
    "❤️": "__emoji_positive__",
    "👍": "__emoji_positive__",
    "🔥": "__emoji_positive__",
    "😢": "__emoji_negative__",
    "😭": "__emoji_negative__",
    "😡": "__emoji_negative__",
    "😤": "__emoji_negative__",
    "💔": "__emoji_negative__",
    "👎": "__emoji_negative__",
    "💀": "__emoji_negative__",
    "🥲": "__emoji_negative__",
}

NEGATION_WORDS = {"not", "never", "no", "dont", "doesnt", "didnt", "wont", "cant", "isnt", "wasnt"}

# Words that signal an inherently unpleasant situation even without emotional language.
# Used to detect sarcasm: a positive emotion word near these likely means the opposite.
NEGATIVE_SITUATION_WORDS = {
    "stuck", "traffic", "fail", "failing", "failed", "broke", "broken",
    "missed", "late", "wrong", "problem", "sick", "hurt", "waiting",
    "wasted", "waste", "lose", "lost", "annoying", "annoyed", "terrible",
    "awful", "horrible", "disaster", "slow", "delayed", "delay",
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        for emoji, signal in TEXT_EMOJI_MAP.items():
            text = text.replace(emoji, f" {signal} ")

        for emoji, signal in UNICODE_EMOJI_MAP.items():
            text = text.replace(emoji, f" {signal} ")

        text = text.strip().lower()

        text = re.sub(r"[^\w\s]", " ", text)

        tokens = text.split()
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        score = 0
        negate = False

        for i, token in enumerate(tokens):
            if token in NEGATION_WORDS:
                negate = True
                continue

            if token == "__emoji_positive__":
                delta = 2
            elif token == "__emoji_negative__":
                delta = -2
            elif token in self.positive_words:
                window = tokens[max(0, i - 4):i] + tokens[i + 1:i + 5]
                if any(w in NEGATIVE_SITUATION_WORDS for w in window):
                    delta = -1
                else:
                    delta = 1
            elif token in self.negative_words:
                delta = -1
            else:
                negate = False
                continue

            score += -delta if negate else delta
            negate = False

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
