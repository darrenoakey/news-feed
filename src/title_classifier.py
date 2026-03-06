import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from src.classifier_backend import (
    TreeBackend,
    LABELS,
    LABEL_SCORES,
    SCORE_THRESHOLDS,
    label_from_score,
)

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 100

# re-export for any code importing from here
__all__ = ["LABELS", "LABEL_SCORES", "SCORE_THRESHOLDS", "label_from_score", "TitleClassifierService"]


# ##################################################################
# title classifier service
# wraps the sklearn decision tree classifier for predicting article quality
class TitleClassifierService:
    def __init__(self):
        self._metrics: Optional[dict] = None
        self._last_trained: Optional[datetime] = None
        self._training_count: int = 0
        # rolling accuracy: tracks (predicted_label, actual_label) for recent classifications
        self._rolling: deque[tuple[str, str]] = deque(maxlen=ROLLING_WINDOW)
        # accuracy history: cumulative accuracy at each vote for charting
        self._accuracy_history: list[float] = []
        self._correct_count: int = 0
        self._total_count: int = 0
        # the sklearn decision tree classifier
        self._backend: Optional[TreeBackend] = None

    # ##################################################################
    # load model
    # attempt to load the sklearn tree model; returns True if successful
    def load_model(self) -> bool:
        backend = TreeBackend()
        if backend.available:
            self._backend = backend
            logger.info("Loaded tree classifier: %s", backend.info())
            return True
        logger.info("Tree classifier not available (no model file)")
        return False

    # ##################################################################
    # record training count
    # update the count of human-labeled samples used for metrics display
    def set_training_count(self, count: int):
        self._training_count = count
        self._last_trained = datetime.now(timezone.utc)
        self._update_metrics()

    # ##################################################################
    # record result
    # record a prediction vs actual for rolling accuracy
    def record_result(self, predicted_label: str, actual_label: str):
        self._rolling.append((predicted_label, actual_label))
        self._total_count += 1
        if predicted_label == actual_label:
            self._correct_count += 1
        self._accuracy_history.append(self._correct_count / self._total_count)
        self._update_metrics()

    # ##################################################################
    # update metrics
    # recompute metrics from rolling window
    def _update_metrics(self):
        rolling = self._rolling_stats()
        backend_info = self._backend.info() if self._backend else None
        self._metrics = {
            "training_samples": self._training_count,
            "last_trained": self._last_trained.isoformat() if self._last_trained else None,
            "rolling": rolling,
            "history": self._accuracy_history,
            "backend": backend_info,
        }

    # ##################################################################
    # rolling stats
    # compute accuracy and per-class stats from rolling window
    def _rolling_stats(self) -> dict:
        if not self._rolling:
            return {"total": 0, "accuracy": 0, "per_class": {}}
        total = len(self._rolling)
        correct = sum(1 for p, a in self._rolling if p == a)
        # per-class precision/recall
        per_class = {}
        for label in LABELS:
            predicted_as = [(p, a) for p, a in self._rolling if p == label]
            actually_is = [(p, a) for p, a in self._rolling if a == label]
            tp = sum(1 for p, a in self._rolling if p == label and a == label)
            precision = tp / len(predicted_as) if predicted_as else 0
            recall = tp / len(actually_is) if actually_is else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "count": len(actually_is)}
        return {"total": total, "accuracy": correct / total, "per_class": per_class}

    def _rolling_accuracy_str(self) -> str:
        if not self._rolling:
            return "no data"
        total = len(self._rolling)
        correct = sum(1 for p, a in self._rolling if p == a)
        return f"{correct}/{total} ({correct/total:.0%})"

    # ##################################################################
    # predict
    # return predicted label and score from the tree classifier
    def predict(self, title: str) -> Optional[tuple[str, float]]:
        if self._backend is None:
            return None
        try:
            return self._backend.classify(title)
        except Exception as err:
            logger.warning("Tree classify failed: %s", err)
            return None

    # ##################################################################
    # get metrics
    # return current cached metrics or None
    def get_metrics(self) -> Optional[dict]:
        return self._metrics

    # ##################################################################
    # is trained
    # return True if the tree classifier is loaded
    @property
    def is_trained(self) -> bool:
        return self._backend is not None

    @property
    def training_count(self) -> int:
        return self._training_count
