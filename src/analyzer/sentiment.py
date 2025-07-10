"""
Advanced sentiment analysis engine with multilingual support, context awareness,
sarcasm detection, aspect-level analysis, and visual reporting.
"""

import re
import logging
import os
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict, Counter
import concurrent.futures
import json
from functools import lru_cache

# Optional imports with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download necessary NLTK resources if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Typing for language codes
LanguageCode = str  # ISO 639-1/2 codes like 'en', 'fr', 'zh', etc.


class SentimentPolarity(Enum):
    """Sentiment polarity classifications."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

    @classmethod
    def from_score(cls, score: float) -> "SentimentPolarity":
        """Convert a normalized score (-1 → 1) to a polarity bucket."""
        if score <= -0.7:
            return cls.VERY_NEGATIVE
        if score <= -0.2:
            return cls.NEGATIVE
        if score <= 0.2:
            return cls.NEUTRAL
        if score <= 0.7:
            return cls.POSITIVE
        return cls.VERY_POSITIVE

    def to_simple(self) -> str:
        """Map to positive / neutral / negative."""
        if self in (self.NEGATIVE, self.VERY_NEGATIVE):
            return "negative"
        if self in (self.POSITIVE, self.VERY_POSITIVE):
            return "positive"
        return "neutral"

    @property
    def emoji(self) -> str:
        return {
            self.VERY_NEGATIVE: "😠",
            self.NEGATIVE: "😞",
            self.NEUTRAL: "😐",
            self.POSITIVE: "🙂",
            self.VERY_POSITIVE: "😄",
        }[self]


@dataclass
class AspectSentiment:
    """Sentiment attached to a specific aspect / entity."""
    aspect: str
    polarity: SentimentPolarity
    score: float
    confidence: float
    text_span: Optional[str] = None
    span_position: Optional[Tuple[int, int]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect,
            "polarity": self.polarity.name,
            "polarity_simple": self.polarity.to_simple(),
            "score": self.score,
            "confidence": self.confidence,
            "text_span": self.text_span,
            "span_position": self.span_position,
        }


@dataclass
class SentimentResult:
    """Full sentiment payload for one text."""
    polarity: SentimentPolarity
    score: float
    confidence: float
    text: str
    language: Optional[LanguageCode] = None

    # extras
    aspects: List[AspectSentiment] = None
    is_sarcastic: Optional[bool] = None
    sarcasm_confidence: Optional[float] = None
    subjectivity: Optional[float] = None
    context_aware: bool = False
    model_name: Optional[str] = None
    processing_time: Optional[float] = None
    positive_phrases: Optional[List[str]] = None
    negative_phrases: Optional[List[str]] = None
    key_phrases: Optional[List[str]] = None
    word_scores: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.aspects is None:
            self.aspects = []

    # Convenience
    @property
    def sentiment(self) -> str:
        return self.polarity.to_simple()

    @property
    def emoji(self) -> str:
        return self.polarity.emoji

    # Serialization
    def as_dict(self) -> Dict[str, Any]:
        base = {
            "sentiment": self.sentiment,
            "polarity": self.polarity.name,
            "score": self.score,
            "confidence": self.confidence,
            "language": self.language,
            "context_aware": self.context_aware,
            "model_name": self.model_name,
            "processing_time": self.processing_time,
        }
        if self.aspects:
            base["aspects"] = [a.as_dict() for a in self.aspects]
        if self.is_sarcastic is not None:
            base["is_sarcastic"] = self.is_sarcastic
            base["sarcasm_confidence"] = self.sarcasm_confidence
        if self.subjectivity is not None:
            base["subjectivity"] = self.subjectivity
        if self.positive_phrases:
            base["positive_phrases"] = self.positive_phrases
        if self.negative_phrases:
            base["negative_phrases"] = self.negative_phrases
        if self.key_phrases:
            base["key_phrases"] = self.key_phrases
        return base

    def visualize(self, title: str = "Sentiment Analysis"):
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available")
            return None

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(title, fontsize=16)

        ax1 = fig.add_subplot(221)
        self._plot_sentiment_gauge(ax1)

        if self.aspects:
            ax2 = fig.add_subplot(222)
            self._plot_aspect_sentiment(ax2)

        if WORDCLOUD_AVAILABLE and self.word_scores:
            ax3 = fig.add_subplot(223)
            self._plot_wordcloud(ax3)

        ax4 = fig.add_subplot(224)
        self._plot_misc(ax4)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    # Internal plotting helpers
    def _plot_sentiment_gauge(self, ax):
        norm_score = (self.score + 1) / 2
        cmap, norm = cm.RdYlGn, Normalize(0, 1)
        ax.barh(0, norm_score, height=0.3, color=cmap(norm(norm_score)))
        ax.barh(0, 1, height=0.3, color="lightgrey", alpha=0.3)
        for x, lab in [(0, "Negative"), (0.5, "Neutral"), (1, "Positive")]:
            ax.axvline(x, color="grey", ls="--", alpha=0.6)
            ax.text(x, -0.2, lab, ha="center", va="top")
        ax.scatter(norm_score, 0, s=150, color="darkblue", zorder=5)
        ax.text(norm_score, 0.1, f"{self.score:.2f}", ha="center", va="bottom", weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_axis_off()
        ax.set_title("Overall Sentiment")

    def _plot_aspect_sentiment(self, ax):
        aspects = sorted(self.aspects, key=lambda a: a.score)
        names = [a.aspect for a in aspects]
        vals = [a.score for a in aspects]
        cmap, norm = cm.RdYlGn, Normalize(-1, 1)
        colors = [cmap(norm(v)) for v in vals]
        y = range(len(names))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.axvline(0, color="grey", alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Score")
        ax.set_title("Aspect-Level Sentiment")

    def _plot_wordcloud(self, ax):
        def color_func(word, *_):
            v = (self.word_scores.get(word, 0) + 1) / 2
            return cm.RdYlGn(v)
        wc = WordCloud(width=800, height=400, background_color="white",
                       color_func=color_func, max_words=100)
        wc.generate_from_frequencies({w: abs(s) + 0.1 for w, s in self.word_scores.items()})
        ax.imshow(wc, interpolation="bilinear")
        ax.set_axis_off()
        ax.set_title("Sentiment Word Cloud")

    def _plot_misc(self, ax):
        rows = []
        if self.subjectivity is not None:
            rows.append(("Subjectivity", f"{self.subjectivity:.2f}"))
        if self.is_sarcastic is not None:
            rows.append(("Sarcasm", f"{self.sarcasm_confidence:.2f}" if self.is_sarcastic else "No"))
        rows.append(("Confidence", f"{self.confidence:.2f}"))
        if self.language:
            rows.append(("Language", self.language))
        if self.processing_time:
            rows.append(("Time (ms)", f"{self.processing_time*1000:.1f}"))
        if self.aspects:
            rows.append(("Aspects", str(len(self.aspects))))
        ax.axis("off")
        if rows:
            table = ax.table(cellText=[[v] for _, v in rows],
                             rowLabels=[k for k, _ in rows],
                             colLabels=["Value"], loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.4)
        ax.set_title("Metrics")

    def generate_text_summary(self) -> str:
        lines = [
            f"Overall sentiment: {self.polarity.name} ({self.score:.2f}) "
            f"with {self.confidence:.2f} confidence."
        ]
        if self.is_sarcastic is not None:
            lines.append(
                "Sarcasm detected." if self.is_sarcastic
                else "No sarcasm detected."
            )
        if self.language:
            lines.append(f"Language: {self.language}")
        if self.aspects:
            lines.append("Aspect-level sentiment:")
            for a in self.aspects:
                lines.append(f"  • {a.aspect}: {a.polarity.name} ({a.score:.2f})")
        if self.processing_time:
            lines.append(f"Processed in {self.processing_time*1000:.1f} ms.")
        return "\n".join(lines)


class SentimentEngine:
    DEFAULT_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
    DEFAULT_ASPECT_MODEL = DEFAULT_SENTIMENT_MODEL

    # Hand-crafted sarcasm regexes
    _SARCASM_PATTERNS = [
        r"yeah right", r"oh sure", r"\bwhatever\b", r"/s\b",
        r"🙄", r"😏", r"as if", r"i bet", r"totally", r"big surprise"
    ]

    def __init__(
        self,
        sentiment_model: Optional[str] = None,
        aspect_model: Optional[str] = None,
        detect_language: bool = True,
        detect_sarcasm: bool = True,
        detect_aspects: bool = True,
        cache_size: int = 2048,
        max_workers: int = 4,
        device: Optional[str] = None,
    ):
        self.sentiment_model_name = sentiment_model or self.DEFAULT_SENTIMENT_MODEL
        self.aspect_model_name = aspect_model or self.DEFAULT_ASPECT_MODEL
        self.detect_language = detect_language
        self.detect_sarcasm = detect_sarcasm
        self.detect_aspects = detect_aspects
        self.max_workers = max_workers
        self.device = device or self._best_device()

        self._sarcasm_regexes = [re.compile(pat, re.I) for pat in self._SARCASM_PATTERNS]
        self._init_models()

        # LRU cache wrapper
        self.analyze = lru_cache(maxsize=cache_size)(self._analyze_uncached)

    def _best_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _init_models(self):
        from transformers import pipeline
        logging.info("Loading sentiment model …")
        self.sentiment_pipe = pipeline("sentiment-analysis",
                                       model=self.sentiment_model_name,
                                       tokenizer=self.sentiment_model_name,
                                       device=self.device)
        if self.detect_aspects:
            if self.aspect_model_name == self.sentiment_model_name:
                self.aspect_pipe = self.sentiment_pipe
            else:
                logging.info("Loading aspect model …")
                self.aspect_pipe = pipeline("sentiment-analysis",
                                             model=self.aspect_model_name,
                                             tokenizer=self.aspect_model_name,
                                             device=self.device)
        if self.detect_language:
            try:
                import fasttext
                if not os.path.exists("lid.176.bin"):
                    import fasttext.util
                    fasttext.util.download_model("lid.176", if_exists="ignore")
                self.lang_model = fasttext.load_model("lid.176.bin")
            except Exception:
                self.lang_model = None
                logging.warning("fastText language model unavailable – "
                                "language detection disabled")
                self.detect_language = False

    def _language(self, text: str) -> Optional[str]:
        if not self.detect_language or not self.lang_model:
            return None
        pred = self.lang_model.predict(text.replace("\n", " "))
        return pred[0][0].replace("__label__", "")

    def _sarcasm(self, text: str) -> Tuple[bool, float]:
        if not self.detect_sarcasm:
            return (False, 0.0)
        score = sum(bool(r.search(text)) for r in self._sarcasm_regexes) / len(self._sarcasm_regexes)
        return (score > 0.3, score)

    def _score_from_label(self, label: str, raw_score: float) -> float:
        lab = label.lower()
        if "positive" in lab:
            return raw_score
        if "negative" in lab:
            return -raw_score
        if "neutral" in lab:
            return 0.0
        if "star" in lab:
            try:
                stars = int(lab.split()[0]); return (stars - 3) / 2
            except ValueError:
                return 0.0
        return 0.0

    def _analyze_uncached(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult(SentimentPolarity.NEUTRAL, 0.0, 1.0, text)

        t0 = time.time()
        lang = self._language(text)

        # overall sentiment
        raw = self.sentiment_pipe(text)[0]
        score = self._score_from_label(raw["label"], raw["score"])
        polarity = SentimentPolarity.from_score(score)

        res = SentimentResult(
            polarity=polarity,
            score=score,
            confidence=raw["score"],
            text=text,
            language=lang,
            model_name=self.sentiment_model_name,
        )

        # sarcasm flip
        sarc, sarc_c = self._sarcasm(text)
        res.is_sarcastic, res.sarcasm_confidence = sarc, sarc_c
        if sarc and sarc_c > 0.7:
            res.score = -res.score
            res.polarity = SentimentPolarity.from_score(res.score)

        # aspects
        if self.detect_aspects:
            aspects = self._extract_aspects(text)
            res.aspects = [self._aspect_sent(text, a) for a in aspects[:10]]

        res.processing_time = time.time() - t0
        return res

    def _extract_aspects(self, text: str) -> List[str]:
        # naive noun-phrase grab if NLTK present, else fallback
        if not NLTK_AVAILABLE:
            words = text.lower().split()
            return list({w for w in words if len(w) > 3})[:10]
        from nltk import pos_tag, word_tokenize, RegexpParser
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        parser = RegexpParser(r"NP: {<JJ>*<NN.*>+}")
        tree = parser.parse(tagged)
        aspects = [' '.join(w for w, _ in n.leaves())
                   for n in tree if hasattr(n, 'label') and n.label() == 'NP']
        return aspects

    def _aspect_sent(self, text: str, aspect: str) -> AspectSentiment:
        idx = text.lower().find(aspect.lower())
        window = text[max(0, idx - 80): idx + len(aspect) + 80]
        raw = self.aspect_pipe(window)[0]
        score = self._score_from_label(raw["label"], raw["score"])
        return AspectSentiment(
            aspect=aspect,
            polarity=SentimentPolarity.from_score(score),
            score=score,
            confidence=raw["score"],
            text_span=window.strip(),
            span_position=(idx, idx + len(aspect)),
        )

    def analyze_batch(self, texts: List[str],
                      progress: Optional[Callable[[int, int], None]] = None) -> List[SentimentResult]:
        out: List[Optional[SentimentResult]] = [None]*len(texts)
        with concurrent.futures.ThreadPoolExecutor(self.max_workers) as ex:
            futs = {ex.submit(self.analyze, t): i for i, t in enumerate(texts)}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                out[i] = fut.result()
                done += 1
                if progress:
                    progress(done, len(texts))
        return out  # type: ignore

    def analyze_file(self, path: str, out_json: Optional[str] = None) -> Dict[str, Any]:
        lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        results = self.analyze_batch(lines)
        if out_json:
            json.dump([r.as_dict() for r in results], open(out_json, "w", encoding="utf-8"), indent=2)
        return {
            "total": len(results),
            "avg_score": sum(r.score for r in results)/len(results) if results else 0,
            "sentiments": Counter(r.sentiment for r in results)
        }


def visualize_sentiment_comparison(results: List[SentimentResult], title="Sentiment Comparison"):
    if not MATPLOTLIB_AVAILABLE or not results:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ys = range(len(results))
    scores = [r.score for r in results]
    colors = ['green' if s > 0.2 else 'red' if s < -0.2 else 'blue' for s in scores]
    ax.barh(ys, scores, color=colors)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"Text {i+1}" for i in ys])
    ax.axvline(0, color="grey", alpha=0.4)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Score (-1 → 1)")
    ax.set_title(title)
    return fig


def calculate_sentiment_agreement(results: List[SentimentResult]) -> float:
    if len(results) < 2:
        return 1.0
    counts = Counter(r.sentiment for r in results)
    return counts.most_common(1)[0][1] / len(results)
