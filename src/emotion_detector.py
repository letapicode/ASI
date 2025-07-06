from __future__ import annotations

"""Simple emotion detection using TextBlob sentiment analysis."""

from textblob import TextBlob

__all__ = ["detect_emotion"]


def detect_emotion(text: str) -> str:
    """Return the dominant emotion of ``text``.

    The function categorizes polarity from :class:`textblob.blob.TextBlob`
    into ``"positive"``, ``"neutral"`` or ``"negative"``. The threshold is
    deliberately small so that mildly sentimental phrases count as neutral.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    if polarity < -0.2:
        return "negative"
    return "neutral"
