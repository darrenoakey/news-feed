import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib  # pyright: ignore[reportMissingImports]
from sklearn.feature_extraction.text import CountVectorizer  # pyright: ignore[reportMissingImports]
from sklearn.tree import DecisionTreeClassifier  # pyright: ignore[reportMissingImports]

from src.classifier_trainer import evaluate, ERROR_WEIGHTS, train_tree


# ##################################################################
# test evaluate perfect classifier
def test_evaluate_perfect():
    def perfect_classify(title):
        if "breakthrough" in title:
            return ("great", 2.0)
        if "update" in title:
            return ("good", 1.0)
        return ("other", 0.0)

    labeled = {
        "great": ["breakthrough in AI"],
        "good": ["minor update released"],
        "other": ["boring press release"],
    }
    errors, counts = evaluate(perfect_classify, labeled)
    assert errors == []
    assert counts["accuracy"] == 1.0
    assert counts["correct"] == 3
    assert counts["total"] == 3


# ##################################################################
# test evaluate with errors
def test_evaluate_with_errors():
    def bad_classify(title):
        return ("other", 0.0)  # always predicts other

    labeled = {
        "great": ["breakthrough in AI"],
        "good": ["minor update released"],
        "other": ["boring press release"],
    }
    errors, counts = evaluate(bad_classify, labeled)
    assert len(errors) == 2  # great->other and good->other
    assert counts["correct"] == 1  # only "other" is correct
    assert counts["accuracy"] == 1 / 3


# ##################################################################
# test error weights are applied correctly
def test_error_weights():
    def swap_classify(title):
        if "breakthrough" in title:
            return ("other", 0.0)  # great->other = weight 50
        if "update" in title:
            return ("great", 2.0)  # good->great = weight 5
        return ("other", 0.0)

    labeled = {
        "great": ["breakthrough in AI"],
        "good": ["update released"],
        "other": ["boring press release"],
    }
    errors, counts = evaluate(swap_classify, labeled)
    assert len(errors) == 2
    # errors sorted by weight descending
    assert errors[0]["weight"] == 50  # great->other
    assert errors[1]["weight"] == 5   # good->great


# ##################################################################
# test error weights match specification
def test_error_weight_values():
    assert ERROR_WEIGHTS[("great", "other")] == 50
    assert ERROR_WEIGHTS[("good", "other")] == 50
    assert ERROR_WEIGHTS[("other", "great")] == 30
    assert ERROR_WEIGHTS[("other", "good")] == 10
    assert ERROR_WEIGHTS[("great", "good")] == 5
    assert ERROR_WEIGHTS[("good", "great")] == 5


# ##################################################################
# test evaluate handles exceptions in classifier
def test_evaluate_handles_classifier_error():
    def broken_classify(title):
        raise ValueError("broken")

    labeled = {"great": ["test title"], "good": [], "other": []}
    errors, counts = evaluate(broken_classify, labeled)
    assert len(errors) == 1
    assert errors[0]["predicted"] == "error"
    assert errors[0]["weight"] == 50


# ##################################################################
# test evaluate per-class stats
def test_evaluate_per_class_stats():
    def mostly_right(title):
        if "great" in title:
            return ("great", 2.0)
        if "good" in title:
            return ("good", 1.0)
        return ("other", 0.0)

    labeled = {
        "great": ["great news", "great discovery"],
        "good": ["good update"],
        "other": ["other stuff", "other things"],
    }
    errors, counts = evaluate(mostly_right, labeled)
    assert counts["by_class"]["great"]["total"] == 2
    assert counts["by_class"]["great"]["correct"] == 2
    assert counts["by_class"]["good"]["total"] == 1
    assert counts["by_class"]["good"]["correct"] == 1
    assert counts["by_class"]["other"]["total"] == 2
    assert counts["by_class"]["other"]["correct"] == 2


# ##################################################################
# test evaluate top errors capped at 100
def test_evaluate_caps_errors():
    def always_wrong(title):
        return ("other", 0.0)

    labeled = {"great": [f"title {i}" for i in range(150)], "good": [], "other": []}
    errors, counts = evaluate(always_wrong, labeled)
    assert len(errors) == 100  # capped at TOP_ERRORS
    assert counts["total"] == 150


# ##################################################################
# test train_tree produces a working model
def test_train_tree_produces_model():
    labeled = {
        "great": ["amazing breakthrough in quantum computing", "revolutionary AI model released", "groundbreaking new chip design", "world first fusion reactor achieved"],
        "good": ["minor software update available", "new tutorial on Python basics", "interesting guide to machine learning", "how to build web apps quickly"],
        "other": ["company hires new CEO", "quarterly earnings report released", "boring press release today", "annual shareholder meeting scheduled"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"
        with patch("src.classifier_trainer.export_labeled_data", return_value=labeled), \
             patch("src.classifier_trainer.MODEL_PATH", model_path):
            train_tree()

        assert model_path.exists()
        vectorizer, tree = joblib.load(model_path)
        assert isinstance(vectorizer, CountVectorizer)
        assert isinstance(tree, DecisionTreeClassifier)

        # verify 100% on training data
        all_titles = labeled["great"] + labeled["good"] + labeled["other"]
        all_labels = ["great"] * 4 + ["good"] * 4 + ["other"] * 4
        X = vectorizer.transform(all_titles)
        predictions = tree.predict(X)
        assert list(predictions) == all_labels


# ##################################################################
# test train_tree skips with insufficient data
def test_train_tree_insufficient_data(capsys):
    labeled = {"great": ["one"], "good": [], "other": []}
    with patch("src.classifier_trainer.export_labeled_data", return_value=labeled):
        train_tree()
    output = capsys.readouterr().out
    assert "Not enough labeled data" in output
