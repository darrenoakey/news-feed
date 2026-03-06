import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "local" / "classifier_model.joblib"

# error weights: how bad is each misclassification?
ERROR_WEIGHTS = {
    ("great", "other"): 50,   # great called other — worst
    ("good", "other"): 50,    # good called other — worst
    ("other", "great"): 30,   # other called great — bad spam
    ("other", "good"): 10,    # other called good — mild spam
    ("great", "good"): 5,     # great/good confused — minor
    ("good", "great"): 5,     # great/good confused — minor
}

TOP_ERRORS = 100


# ##################################################################
# export labeled data
# load all human-labeled titles from the database grouped by label
def export_labeled_data() -> dict[str, list[str]]:
    from src.database import get_session
    from src.models import TitleClassification

    # deduplicate by title, keeping the most recent label
    title_to_label: dict[str, str] = {}
    with get_session() as session:
        labeled = (
            session.query(TitleClassification)
            .filter(TitleClassification.human_label.isnot(None))
            .order_by(TitleClassification.classified_at.asc())
            .all()
        )
        for tc in labeled:
            if tc.human_label in ("great", "good", "other"):
                title_to_label[tc.title] = tc.human_label

    by_label: dict[str, list[str]] = {"great": [], "good": [], "other": []}
    for title, label in title_to_label.items():
        by_label[label].append(title)
    return by_label


# ##################################################################
# evaluate classifier
# run all labeled titles through classifier, return weighted errors
def evaluate(classify_fn, labeled: dict[str, list[str]]) -> tuple[list[dict], dict]:
    errors = []
    counts = {"total": 0, "correct": 0, "by_class": {}}

    for actual_label, titles in labeled.items():
        for title in titles:
            counts["total"] += 1
            try:
                predicted_label, score = classify_fn(title)
            except Exception as err:
                errors.append({
                    "title": title,
                    "actual": actual_label,
                    "predicted": "error",
                    "score": 0.0,
                    "weight": 50,
                    "error": str(err),
                })
                continue

            if predicted_label == actual_label:
                counts["correct"] += 1
            else:
                weight = ERROR_WEIGHTS.get((actual_label, predicted_label), 1)
                errors.append({
                    "title": title,
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "score": score,
                    "weight": weight,
                })

            # track per-class accuracy
            key = actual_label
            if key not in counts["by_class"]:
                counts["by_class"][key] = {"total": 0, "correct": 0}
            counts["by_class"][key]["total"] += 1
            if predicted_label == actual_label:
                counts["by_class"][key]["correct"] += 1

    # sort by weight descending, take top N
    errors.sort(key=lambda e: e["weight"], reverse=True)
    accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
    counts["accuracy"] = accuracy
    return errors[:TOP_ERRORS], counts


# ##################################################################
# train tree
# train a DecisionTreeClassifier on labeled data and save to disk
def train_tree():
    labeled = export_labeled_data()
    total = sum(len(v) for v in labeled.values())
    print(f"Labeled data: {total} titles — great={len(labeled['great'])}, good={len(labeled['good'])}, other={len(labeled['other'])}")

    if total < 10:
        print("Not enough labeled data to train. Label more titles first.")
        return

    # build training data
    titles = []
    labels = []
    for label in ("great", "good", "other"):
        for title in labeled[label]:
            titles.append(title)
            labels.append(label)

    # vectorize: binary word unigrams + bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True)
    X = vectorizer.fit_transform(titles)

    # train decision tree — no depth limit → 100% on training data
    tree = DecisionTreeClassifier(criterion="entropy")
    tree.fit(X, labels)

    # verify 100% on training data
    predictions = tree.predict(X)
    correct = sum(1 for p, a in zip(predictions, labels) if p == a)
    accuracy = correct / len(labels)
    print(f"Training accuracy: {accuracy:.1%} ({correct}/{len(labels)})")

    if accuracy < 1.0:
        mismatches = [(t, a, p) for t, a, p in zip(titles, labels, predictions) if a != p]
        for title, actual, predicted in mismatches[:10]:
            print(f"  WRONG: \"{title[:80]}\" actual={actual} predicted={predicted}")
        raise RuntimeError(f"Expected 100% training accuracy, got {accuracy:.1%}")

    # save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((vectorizer, tree), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"  Tree depth: {tree.get_depth()}, leaves: {tree.get_n_leaves()}, features: {len(vectorizer.get_feature_names_out())}")


# ##################################################################
# main
# CLI entry point
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    train_tree()


if __name__ == "__main__":
    main()
