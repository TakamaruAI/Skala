import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skala import SkalaClassifier


def main() -> int:
    default = "I have this Python codebase, can you port it to C++"
    text = " ".join(sys.argv[1:]) or default
    classifier = SkalaClassifier(os.path.join(ROOT, "artifacts"))
    classifier.load()
    classifier.warmup()

    label, scores = classifier.classify(text)
    print(f"{text!r} -> {label}")
    for name in ("simple", "medium", "complex"):
        print(f"  {name}: {scores[name]:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
