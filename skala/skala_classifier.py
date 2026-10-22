import json
import os
from collections.abc import Iterable

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack as shstack

from .tool_struct import TOOL_STRUCT_FEATURE_NAMES, ToolStruct

MAX_TOKENS = 64


def physical_cores() -> int:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            pairs = set()
            physical_id = core_id = None
            for line in stream:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key == "physical id":
                    physical_id = value
                elif key == "core id":
                    core_id = value
                elif (
                    not line.strip()
                    and physical_id is not None
                    and core_id is not None
                ):
                    pairs.add((physical_id, core_id))
                    physical_id = core_id = None
            if physical_id is not None and core_id is not None:
                pairs.add((physical_id, core_id))
            if pairs:
                return len(pairs)
    except OSError:
        pass
    return os.cpu_count() or 1


ONNX_THREADS = max(1, min(4, physical_cores()))


class SkalaClassifier:
    def __init__(
        self,
        artifacts_dir: str,
        should_clamp: bool = True,
        threads: int | None = None,
    ) -> None:
        self._dir = artifacts_dir
        self._should_clamp = should_clamp
        self._threads = threads
        self._labels: tuple[str, ...] = ()
        self._thresholds: np.ndarray | None = None
        self._model = None
        self._calibrators: list = []
        self._vectorizer = None
        self._session = None
        self._tokenizer = None
        self._struct = ToolStruct()
        self._clamp: np.ndarray | None = None
        self._clamp_percentile: int | None = None
        self._name = ""
        self._version = ""
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def clamp_percentile(self) -> int | None:
        return self._clamp_percentile

    def _thread_count(self) -> int:
        if self._threads is not None:
            return max(1, int(self._threads))
        return ONNX_THREADS

    def _path(self, name: str) -> str:
        return os.path.join(self._dir, name)

    def load(self) -> None:
        import onnxruntime as ort
        from transformers import PreTrainedTokenizerFast

        with open(self._path("thresholds.json"), encoding="utf-8") as stream:
            meta = json.load(stream)

        self._name = str(meta.get("name", ""))
        self._version = str(meta.get("version", ""))
        self._labels = tuple(meta["labels"])
        self._thresholds = np.array(meta["thresholds"], dtype=np.float64)
        self._clamp = self._load_clamp(meta.get("clamp", "struct_clamp.json"))

        head = joblib.load(self._path(meta.get("head", "head.joblib")))
        self._model = head["lr"]
        self._calibrators = head["iso"]
        self._vectorizer = head["vw"]

        options = ort.SessionOptions()
        options.intra_op_num_threads = self._thread_count()
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            self._path(meta.get("encoder", "encoder.int8.onnx")),
            options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer_path = self._path(meta.get("tokenizer", "tokenizer.json"))
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        self._is_loaded = True

    def _load_clamp(self, name: str) -> np.ndarray | None:
        path = self._path(name)
        if not self._should_clamp or not os.path.exists(path):
            return None

        with open(path, encoding="utf-8") as stream:
            guard = json.load(stream)

        features = list(guard.get("features", []))
        if features and features != TOOL_STRUCT_FEATURE_NAMES:
            raise ValueError(
                "struct_clamp.json does not match the feature module: "
                f"{features} != {TOOL_STRUCT_FEATURE_NAMES}"
            )

        percentile = guard.get("percentile")
        self._clamp_percentile = int(percentile) if percentile is not None else None

        return np.array(guard["clamp"], dtype=np.float64)

    def warmup(self) -> None:
        self.classify("warmup")

    def classify(self, text: str) -> tuple[str, dict[str, float]]:
        if not self._is_loaded:
            raise RuntimeError("call load() before classify")

        raw = self._model.predict_proba(self._features(text))
        calibrated = np.column_stack(
            [
                self._calibrators[index].predict(raw[:, index])
                for index in range(len(self._calibrators))
            ]
        )

        hits = calibrated >= self._thresholds
        decision = int(calibrated.argmax(1)[0])
        if int(hits.sum()) == 1:
            decision = int(hits[0].argmax())

        scores = {
            label: float(score) for label, score in zip(self._labels, calibrated[0])
        }

        return self._labels[decision], scores

    def classify_batch(
        self, texts: Iterable[str]
    ) -> list[tuple[str, dict[str, float]]]:
        return [self.classify(text) for text in texts]

    def _embed(self, text: str) -> np.ndarray:
        encoded = self._tokenizer(
            [str(text)],
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="np",
        )
        outputs = self._session.run(
            None,
            {
                "ids": encoded["input_ids"].astype(np.int64),
                "mask": encoded["attention_mask"].astype(np.int64),
            },
        )

        return np.asarray(outputs[0], dtype=np.float64)

    def _features(self, text: str) -> csr_matrix:
        struct = self._struct.transform([text]).toarray()
        if self._clamp is not None:
            struct = np.minimum(struct, self._clamp[None, :])

        columns = [
            csr_matrix(self._embed(text)),
            self._vectorizer.transform([text]),
            csr_matrix(struct),
        ]

        return shstack(columns).tocsr()
