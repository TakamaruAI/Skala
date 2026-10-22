import re

FEATURE_NAMES = [
    "n_words",
    "n_sents",
    "n_punct",
    "avg_sent_len",
    "n_code_blocks",
    "uppercase_ratio",
    "n_questions",
    "has_code",
    "has_arch",
    "has_math",
    "has_debug",
    "has_research",
    "n_complex_verbs",
    "n_scale_adjs",
    "n_hard_nouns",
    "has_simple",
    "has_realtime",
    "has_long_output",
    "avg_word_len",
]

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")
_PUNCT_RE = re.compile(r"[^\w\s]")

CODE_PAT = re.compile(
    r"(```|~~~|`[^`]+`|\b(def |class |import |from .* import|function |const |let |"
    r"var |SELECT |INSERT |UPDATE |DELETE |public static|#include|fn main|=>"
    r"|println!|System\.out))"
)
ARCH_WORDS = {
    "architecture",
    "architect",
    "microservice",
    "microservices",
    "distributed",
    "pipeline",
    "infrastructure",
    "kubernetes",
    "cluster",
    "scalab",
    "deploy",
    "migration",
    "monolith",
    "shard",
    "replica",
    "consensus",
    "raft",
    "paxos",
}
MATH_PAT = re.compile(
    r"[=<>+\-*/^√∫∑∏∂∇≤≥≠≈∞]|\b(theorem|proof|prove|lemma|equation|"
    r"induction|derivative|integral|matrix|eigen|probability|qed)\b"
)
DEBUG_WORDS = {
    "debug",
    "debugging",
    "error",
    "bug",
    "traceback",
    "stacktrace",
    "exception",
    "fix",
    "broken",
    "crash",
    "fail",
    "failing",
    "patch",
    "hotfix",
}
RESEARCH_WORDS = {
    "research",
    "paper",
    "study",
    "survey",
    "literature",
    "benchmark",
    "dataset",
    "baseline",
    "ablation",
    "sota",
    "state-of-the-art",
}
COMPLEX_VERBS = {
    "analyze",
    "analyse",
    "design",
    "implement",
    "optimize",
    "optimise",
    "architect",
    "prove",
    "derive",
    "synthesize",
    "evaluate",
    "compare",
    "migrate",
    "deploy",
    "scale",
    "secure",
    "refactor",
    "integrate",
    "orchestrate",
    "formalize",
    "generalize",
    "debug",
}
SCALE_ADJS = {
    "distributed",
    "large-scale",
    "large scale",
    "massive",
    "enterprise",
    "global",
    "high-throughput",
    "high throughput",
    "fault-tolerant",
    "fault tolerant",
    "real-time",
    "realtime",
    "petabyte",
    "million",
    "billion",
    "end-to-end",
    "end to end",
    "production-grade",
    "production grade",
}
HARD_NOUNS = {
    "kubernetes",
    "microservices",
    "redis",
    "postgres",
    "theorem",
    "proof",
    "induction",
    "quantum",
    "compiler",
    "kernel",
    "cryptography",
    "blockchain",
    "consensus",
    "neural",
    "transformer",
    "gradient",
    "eigen",
    "mlops",
    "terraform",
    "kafka",
    "spark",
    "hadoop",
    "docker",
    "inference",
}
SIMPLE_PAT = re.compile(
    r"\b(simple|simply|easy|basic|basics|hello|hi\b|what is|who is|where is"
    r"|how do i say)\b"
)
REALTIME_PAT = re.compile(r"\b(real-?time|live|streaming|latency|deadline|instant)\b")
LONGOUT_PAT = re.compile(
    r"\b(essay|report|thesis|detailed|comprehensive|in-?depth|"
    r"step-?by-?step|thorough|elaborate|paragraphs|pages?)\b"
)


def _count_phrases(text_low: str, phrases: tuple[str, ...]) -> int:
    return sum(text_low.count(phrase) for phrase in phrases)


def featurize(text: str) -> list[float]:
    source = text if isinstance(text, str) else ""
    low = source.lower()
    words = _WORD_RE.findall(source)
    n_words = len(words)
    sentences = [part for part in _SENT_RE.findall(source) if part.strip()]
    n_sents = max(len(sentences), 1)
    n_punct = len(_PUNCT_RE.findall(source))
    avg_sent_len = n_words / n_sents
    n_code_blocks = len(re.findall(r"```|~~~", source)) // 2 + len(
        re.findall(r"(?m)^(?:    |\t)\S", source)
    )
    alpha = [char for char in source if char.isalpha()]
    uppercase_ratio = (
        (sum(char.isupper() for char in alpha) / len(alpha)) if alpha else 0.0
    )
    n_questions = source.count("?")
    has_code = 1.0 if CODE_PAT.search(source) else 0.0
    has_arch = 1.0 if any(word in low for word in ARCH_WORDS) else 0.0
    has_math = 1.0 if MATH_PAT.search(low) else 0.0
    has_debug = 1.0 if any(word in low for word in DEBUG_WORDS) else 0.0
    has_research = 1.0 if any(word in low for word in RESEARCH_WORDS) else 0.0
    n_complex_verbs = float(_count_phrases(low, COMPLEX_VERBS))
    n_scale_adjs = float(_count_phrases(low, SCALE_ADJS))
    n_hard_nouns = float(_count_phrases(low, HARD_NOUNS))
    has_simple = 1.0 if SIMPLE_PAT.search(low) else 0.0
    has_realtime = 1.0 if REALTIME_PAT.search(low) else 0.0
    has_long_output = 1.0 if LONGOUT_PAT.search(low) else 0.0
    avg_word_len = (sum(len(word) for word in words) / n_words) if n_words else 0.0

    return [
        n_words,
        n_sents,
        n_punct,
        avg_sent_len,
        n_code_blocks,
        uppercase_ratio,
        n_questions,
        has_code,
        has_arch,
        has_math,
        has_debug,
        has_research,
        n_complex_verbs,
        n_scale_adjs,
        n_hard_nouns,
        has_simple,
        has_realtime,
        has_long_output,
        avg_word_len,
    ]
