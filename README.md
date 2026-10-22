# Skala

### A prompt-difficulty classifier for simple, medium, and complex model routing.

Skala classifies chat, agent, and tooling prompts into three difficulty tiers to help routers select an appropriate model for each request. Its 23.2 MB model delivers median CPU inference latency as low as 2.65 ms. Skala combines a linear classifier over lexical and structural features with semantic embeddings from a pruned, int8-quantised variant of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), trained on 6,948 English prompts.

## Architecture

- **Architecture:** Hybrid prompt classifier combining semantic embeddings, lexical features, and structural features
- **Encoder:** Pruned, int8-quantised variant of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), with 5 Transformer layers and 20.9M parameters
- **Encoder configuration:** 12 attention heads, 384-dimensional embeddings, 1,536-dimensional feed-forward layers
- **Feature blocks:** 384 semantic features, 15,000 lexical features, and 21 structural features
- **Classification head:** Multinomial logistic regression, 3 classes
- **Calibration:** Per-class isotonic regression, fitted out-of-fold
- **Decision rule:** Highest calibrated class score
- **Length guard:** Structural features clamped to their training-set 95th percentiles before scoring
- **Maximum sequence length:** 64 tokens

## Datasets

Skala was trained on 6,948 English prompts, 1,698 sourced from the [Nasim435/Multi-label-Prompt-Dataset](https://huggingface.co/datasets/Nasim435/Multi-label-Prompt-Dataset) and 5,250 generated and labelled by two open-weight models, `google/gemma-4-26B-A4B` and `Qwen/Qwen3.8-27B`.

| dataset | rows | licence |
|---|---:|---|
| [Nasim435/Multi-label-Prompt-Dataset](https://huggingface.co/datasets/Nasim435/Multi-label-Prompt-Dataset) | 1,698 | MIT |
| [data/generated_prompts.csv](data/generated_prompts.csv) | 5,242 | MIT |
| [data/real_350.csv](data/real_350.csv) | 350 | MIT |
| [tests/behavior_suite2.csv](tests/behavior_suite2.csv) | 52 | MIT |
| [tests/behavior_suite2_clean.csv](tests/behavior_suite2_clean.csv) | 38 | MIT |
| [tests/holdout_150.csv](tests/holdout_150.csv) | 150 | MIT |

## Results

Evaluation covers behavioural test suites and held-out prompts. The 350-prompt set was evaluated using out-of-fold predictions, where each prompt was scored by a model that had not been trained on it. The 150-prompt holdout was kept separate from training and used for final evaluation.

| Benchmark | n | Score |
|---|---:|---:|
| [tests/behavior_suite2.csv](tests/behavior_suite2.csv) | 52 | **44/52 (84.6%)** |
| [tests/behavior_suite2_clean.csv](tests/behavior_suite2_clean.csv) | 38 | **31/38 (81.6%)** |
| [data/real_350.csv](data/real_350.csv) | 350 | **0.7429** |
| [tests/holdout_150.csv](tests/holdout_150.csv) | 150 | **116/150 (77.3%)** |

## Performance

Skala delivers low-latency, single-prompt inference across desktop and mobile-class CPUs.

| Hardware | p50 (median) | p95 | Calls |
|---|---:|---:|---:|
| Intel Core i7-10700K | 2.75 ms | 3.69 ms | 200 |
| Intel Core i5-4460 | 3.79 ms | 4.44 ms | 200 |
| Snapdragon 8 Gen 3 (SM8650) | 2.65 ms | 3.96 ms | 200 |

All measurements use single-prompt inference (batch size 1), with one warmup pass.

### Runtime footprint

- **Model size:** 23.2 MB
- **Peak memory usage:** ~289 MB RSS

## Limitations

- **English only.** Non-English prompts may be misclassified as `simple` without warning.
- **Length bias.** Longer prompts can lean toward `medium`, even when the task itself is simple.
- **Evaluation caveats.** 14 of 52 behavioural-suite prompts overlap with training data. The clean subset scores 31/38 (81.6%). The separate 150-prompt holdout scores 116/150 (77.3%). Its labels are public and should remain excluded from future training.
- **Single-prompt inference.** Batch classification currently processes prompts one at a time.
- **Retraining variability.** Results may shift slightly between training runs. Published scores refer to the released model artifact.

## Requirements

##### Core dependencies:

- [scikit-learn](https://github.com/scikit-learn/scikit-learn) (pinned to 1.9.1: the fitted head will not unpickle under other versions)
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [joblib](https://github.com/joblib/joblib)
- [onnxruntime](https://github.com/microsoft/onnxruntime)
- [transformers](https://github.com/huggingface/transformers) (tokenizer only, no torch)

##### Python:

- 3.12+

## Installation

Install with:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

## Usage

```python
import sys
sys.path.insert(0, ".")

from skala import SkalaClassifier

clf = SkalaClassifier("artifacts")
clf.load()
clf.warmup()

# Single prompt
label, scores = clf.classify("I have this Python codebase, can you port it to C++")
print(label, scores)

# Several prompts at once
pairs = clf.classify_batch([
    "hi",
    "fix my wifi",
    "prove the Riemann hypothesis",
])
```

## Examples

Classify a prompt from the command line or reproduce the suite numbers from this folder alone

```bash
python examples/classify_prompt.py "I have this Python codebase, can you port it to C++"
python tests/test_suite.py
```

## License

Skala is licensed under the [MIT License](LICENSE).