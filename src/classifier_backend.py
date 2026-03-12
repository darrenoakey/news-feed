import logging
from pathlib import Path
from typing import Protocol

import joblib  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

LABELS = ["great", "good", "other"]
LABEL_SCORES = {"great": 2.0, "good": 1.0, "other": 0.0}
SCORE_THRESHOLDS = {"great": 1.5, "good": 0.5}

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "local" / "classifier_model.joblib"
SVM_MODEL_PATH = PROJECT_ROOT / "local" / "svm_model.joblib"


# ##################################################################
# label from score
# convert a 0-2 score to a label using thresholds
def label_from_score(score: float) -> str:
    if score >= SCORE_THRESHOLDS["great"]:
        return "great"
    elif score >= SCORE_THRESHOLDS["good"]:
        return "good"
    return "other"


# ##################################################################
# classifier backend protocol
# any backend must implement classify() and info()
class ClassifierBackend(Protocol):
    def classify(self, title: str) -> tuple[str, float]: ...
    def info(self) -> dict: ...


# ##################################################################
# tree backend
# loads a sklearn DecisionTreeClassifier from a joblib file
class TreeBackend:
    def __init__(self):
        self._vectorizer = None
        self._tree = None
        self.reload()

    def reload(self):
        try:
            vectorizer, tree = joblib.load(MODEL_PATH)
            self._vectorizer = vectorizer
            self._tree = tree
            logger.info("Loaded tree backend: depth=%d, leaves=%d", tree.get_depth(), tree.get_n_leaves())
        except Exception as err:
            logger.warning("Failed to load tree model from %s: %s", MODEL_PATH, err)
            self._vectorizer = None
            self._tree = None

    @property
    def available(self) -> bool:
        return self._tree is not None

    def classify(self, title: str) -> tuple[str, float]:
        if self._vectorizer is None or self._tree is None:
            raise RuntimeError("Tree backend not loaded")
        X = self._vectorizer.transform([title])
        label = self._tree.predict(X)[0]
        return (label, LABEL_SCORES[label])

    def info(self) -> dict:
        if self._tree is None:
            return {"name": "tree", "type": "sklearn", "status": "not_loaded"}
        return {
            "name": "tree",
            "type": "sklearn",
            "depth": int(self._tree.get_depth()),
            "leaves": int(self._tree.get_n_leaves()),
            "features": int(len(self._vectorizer.get_feature_names_out())),
        }


# ##################################################################
# SVM backend
# loads a sklearn LinearSVC with TF-IDF vectorizer from a joblib file
class SVMBackend:
    def __init__(self):
        self._vectorizer = None
        self._svm = None
        self.reload()

    def reload(self):
        try:
            vectorizer, svm = joblib.load(SVM_MODEL_PATH)
            self._vectorizer = vectorizer
            self._svm = svm
            logger.info("Loaded SVM backend: features=%d", len(vectorizer.get_feature_names_out()))
        except Exception as err:
            logger.warning("Failed to load SVM model from %s: %s", SVM_MODEL_PATH, err)
            self._vectorizer = None
            self._svm = None

    @property
    def available(self) -> bool:
        return self._svm is not None

    def classify(self, title: str) -> tuple[str, float]:
        if self._vectorizer is None or self._svm is None:
            raise RuntimeError("SVM backend not loaded")
        X = self._vectorizer.transform([title])
        label = self._svm.predict(X)[0]
        return (label, LABEL_SCORES[label])

    def info(self) -> dict:
        if self._svm is None:
            return {"name": "svm", "type": "sklearn", "status": "not_loaded"}
        return {
            "name": "svm",
            "type": "sklearn-svm",
            "features": int(len(self._vectorizer.get_feature_names_out())),
        }
