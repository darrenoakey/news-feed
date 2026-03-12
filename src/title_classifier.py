import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from src.classifier_backend import (
    TreeBackend,
    SVMBackend,
    LABELS,
    LABEL_SCORES,
    SCORE_THRESHOLDS,
    label_from_score,
)

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 100

# re-export for any code importing from here
__all__ = ["LABELS", "LABEL_SCORES", "SCORE_THRESHOLDS", "label_from_score", "TitleClassifierService"]

MODEL_NAMES = ("tree", "svm")


# ##################################################################
# per-model accuracy tracker
# tracks rolling accuracy for a single model
class _ModelTracker:
    def __init__(self):
        self.rolling: deque[tuple[str, str]] = deque(maxlen=ROLLING_WINDOW)
        self.accuracy_history: list[float] = []
        self.correct_count: int = 0
        self.total_count: int = 0

    def record(self, predicted_label: str, actual_label: str):
        self.rolling.append((predicted_label, actual_label))
        self.total_count += 1
        if predicted_label == actual_label:
            self.correct_count += 1
        self.accuracy_history.append(self.correct_count / self.total_count)

    def rolling_stats(self) -> dict:
        if not self.rolling:
            return {"total": 0, "accuracy": 0, "per_class": {}}
        total = len(self.rolling)
        correct = sum(1 for p, a in self.rolling if p == a)
        per_class = {}
        for label in LABELS:
            predicted_as = [(p, a) for p, a in self.rolling if p == label]
            actually_is = [(p, a) for p, a in self.rolling if a == label]
            tp = sum(1 for p, a in self.rolling if p == label and a == label)
            precision = tp / len(predicted_as) if predicted_as else 0
            recall = tp / len(actually_is) if actually_is else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "count": len(actually_is)}
        return {"total": total, "accuracy": correct / total, "per_class": per_class}


# ##################################################################
# title classifier service
# wraps both tree and SVM backends for predicting article quality
class TitleClassifierService:
    def __init__(self):
        self._metrics: Optional[dict] = None
        self._last_trained: Optional[datetime] = None
        self._training_count: int = 0
        self._tree_backend: Optional[TreeBackend] = None
        self._svm_backend: Optional[SVMBackend] = None
        self._trackers: dict[str, _ModelTracker] = {
            "tree": _ModelTracker(),
            "svm": _ModelTracker(),
        }

    # ##################################################################
    # load model
    # attempt to load the sklearn tree model; returns True if successful
    def load_model(self) -> bool:
        backend = TreeBackend()
        if backend.available:
            self._tree_backend = backend
            logger.info("Loaded tree classifier: %s", backend.info())
            return True
        logger.info("Tree classifier not available (no model file)")
        return False

    # ##################################################################
    # load svm model
    # attempt to load the SVM model; returns True if successful
    def load_svm_model(self) -> bool:
        backend = SVMBackend()
        if backend.available:
            self._svm_backend = backend
            logger.info("Loaded SVM classifier: %s", backend.info())
            return True
        logger.info("SVM classifier not available (no model file)")
        return False

    # ##################################################################
    # record training count
    def set_training_count(self, count: int):
        self._training_count = count
        self._last_trained = datetime.now(timezone.utc)
        self._update_metrics()

    # ##################################################################
    # record result for a specific model
    def record_result(self, predicted_label: str, actual_label: str, model: str = "tree"):
        tracker = self._trackers.get(model)
        if tracker:
            tracker.record(predicted_label, actual_label)
        self._update_metrics()

    # ##################################################################
    # update metrics
    def _update_metrics(self):
        tree_info = self._tree_backend.info() if self._tree_backend else None
        svm_info = self._svm_backend.info() if self._svm_backend else None
        self._metrics = {
            "training_samples": self._training_count,
            "last_trained": self._last_trained.isoformat() if self._last_trained else None,
            "rolling": self._trackers["tree"].rolling_stats(),
            "history": self._trackers["tree"].accuracy_history,
            "backend": tree_info,
            "svm_rolling": self._trackers["svm"].rolling_stats(),
            "svm_history": self._trackers["svm"].accuracy_history,
            "svm_backend": svm_info,
        }

    # ##################################################################
    # predict using tree backend
    def predict(self, title: str) -> Optional[tuple[str, float]]:
        if self._tree_backend is None:
            return None
        try:
            return self._tree_backend.classify(title)
        except Exception as err:
            logger.warning("Tree classify failed: %s", err)
            return None

    # ##################################################################
    # predict using SVM backend
    def predict_svm(self, title: str) -> Optional[tuple[str, float]]:
        if self._svm_backend is None:
            return None
        try:
            return self._svm_backend.classify(title)
        except Exception as err:
            logger.warning("SVM classify failed: %s", err)
            return None

    # ##################################################################
    # get metrics
    def get_metrics(self) -> Optional[dict]:
        return self._metrics

    # ##################################################################
    # is trained
    @property
    def is_trained(self) -> bool:
        return self._tree_backend is not None

    @property
    def is_svm_trained(self) -> bool:
        return self._svm_backend is not None

    @property
    def training_count(self) -> int:
        return self._training_count
