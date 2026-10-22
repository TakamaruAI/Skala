from collections.abc import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from .structural_features import FEATURE_NAMES, featurize

TOOL_WORDS = (
    "search",
    "look up",
    "lookup",
    "latest",
    "current",
    "today",
    "news",
    "weather",
    "price",
    "stock",
    "score",
    "live",
    "fetch",
    "browse",
    "web",
    "internet",
    "calendar",
    "schedule",
    "concert",
    "trending",
    "recent",
    "up-to-date",
)


TOOL_STRUCT_FEATURE_NAMES = list(FEATURE_NAMES) + ["has_tool", "tool_count"]


class ToolStruct(BaseEstimator, TransformerMixin):
    """19 structural features plus two tool-dependency cues."""

    def fit(self, X: Iterable[str], y: np.ndarray | None = None) -> "ToolStruct":
        return self

    def transform(self, X: Iterable[str]) -> csr_matrix:
        rows = [self._featurize(text) for text in X]

        return csr_matrix(np.asarray(rows, dtype=np.float64))

    @staticmethod
    def _featurize(text: str) -> list[float]:
        source = text if isinstance(text, str) else ""
        hits = sum(source.lower().count(phrase) for phrase in TOOL_WORDS)

        return featurize(source) + [float(hits > 0), float(hits)]
