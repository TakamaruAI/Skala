import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


from skala import SkalaClassifier

EXPECTED_SUITE = (44, 52)
EXPECTED_CLEAN = (31, 38)


def load_rows(name: str) -> list[dict[str, str]]:
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def count_hits(clf: SkalaClassifier, rows: list[dict[str, str]]) -> tuple[int, int]:
    hits = sum(
        clf.classify(row["text"])[0] == row["label"].strip().lower() for row in rows
    )

    return hits, len(rows)


def main() -> int:
    artifacts_dir = os.path.join(os.path.dirname(HERE), "artifacts")
    clf = SkalaClassifier(artifacts_dir=artifacts_dir)
    clf.load()
    clf.warmup()

    hits, total = count_hits(clf, load_rows("behavior_suite2.csv"))
    print(f"suite      {hits}/{total} = {hits / total:.4f}   (expected 44/52)")

    clean_hits, clean_total = count_hits(clf, load_rows("behavior_suite2_clean.csv"))
    print(
        f"clean      {clean_hits}/{clean_total} = {clean_hits / clean_total:.4f}"
        "   (expected 31/38)"
    )

    # The clamp exists so a degenerate long prompt cannot saturate the classifier.
    spam = " ".join(["Number 67"] * 100) + " / 7"
    spam_label, _ = clf.classify(spam)
    print(f"202-word spam -> {spam_label}   (expected medium, not complex)")

    short_label, _ = clf.classify("Pick a number")
    print(f"Pick a number -> {short_label}   (expected simple)")

    is_correct = (
        (hits, total) == EXPECTED_SUITE
        and (clean_hits, clean_total) == EXPECTED_CLEAN
        and spam_label != "complex"
        and short_label == "simple"
    )
    print("\nPASS" if is_correct else "\nFAIL")

    return 0 if is_correct else 1


if __name__ == "__main__":
    sys.exit(main())