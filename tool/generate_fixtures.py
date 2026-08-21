#!/usr/bin/env python3
"""Trains the tiny reference models the test suite checks itself against.

`lid.176` is one model with one loss and no word n-grams, so on its own it
leaves the softmax, negative sampling and one-vs-all paths, and the whole of
`addWordNgrams`, without a reference to compare to. These models fill that
gap: they are small enough to commit, and they are produced by the original
fastText, so the goldens beside them are the source of truth.

The corpus is built so that word order alone separates the labels — `a b a b`
against `b a b a` — which no model can learn from single words. A port that
ignores wordNgrams cannot reproduce these answers.

    python3 -m venv .venv && .venv/bin/pip install fasttext-wheel
    .venv/bin/python tool/generate_fixtures.py
"""
import json
import os
import struct
import sys
import tempfile

import fasttext

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIXTURES = os.path.join(ROOT, "test", "fixtures")
GOLDEN = os.path.join(ROOT, "test", "golden")

# Word order is the only signal: "a b" and "b a" carry the labels.
TRAIN = []
for i in range(60):
    TRAIN.append("__label__ab " + " ".join(["a", "b"] * (2 + i % 3)))
    TRAIN.append("__label__ba " + " ".join(["b", "a"] * (2 + i % 3)))
    TRAIN.append("__label__ab a b " + "c " * (i % 4))
    TRAIN.append("__label__ba b a " + "d " * (i % 4))

TEXTS = [
    "a b a b",
    "b a b a",
    "a b",
    "b a",
    "a a b b",
    "a b c d",
    "c d",
    "a",
    "",
    "   ",
    "__label__ab a b",
    "a b " * 30,
]

# (name, loss, wordNgrams)
MODELS = [
    ("ngrams_hs", "hs", 2),
    ("ngrams_softmax", "softmax", 2),
    ("softmax", "softmax", 1),
    ("negative_sampling", "ns", 1),
    ("one_vs_all", "ova", 1),
]

K = 5


def main() -> int:
    os.makedirs(FIXTURES, exist_ok=True)
    os.makedirs(GOLDEN, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        corpus = os.path.join(work, "train.txt")
        with open(corpus, "w", encoding="utf-8") as handle:
            handle.write("\n".join(TRAIN) + "\n")

        for name, loss, word_ngrams in MODELS:
            model = fasttext.train_supervised(
                input=corpus,
                loss=loss,
                wordNgrams=word_ngrams,
                bucket=100,
                dim=4,
                minn=0,
                maxn=0,
                minCount=1,
                epoch=40,
                lr=0.5,
                seed=1,
                thread=1,
            )
            path = os.path.join(FIXTURES, name + ".bin")
            model.save_model(path)

            cases = []
            for text in TEXTS:
                predictions = model.f.predict(text + "\n", K, 0.0, "strict")
                vector = model.f.getSentenceVector if False else None
                import fasttext_pybind
                vec = fasttext_pybind.Vector(model.get_dimension())
                model.f.getSentenceVector(vec, text + "\n")
                import numpy as np
                packed = b"".join(
                    struct.pack("<f", float(x)) for x in np.asarray(vec)
                )
                cases.append({
                    "text": text,
                    "predictions": [
                        {"label": label.replace("__label__", ""),
                         # Full precision: the value is a float32, so the
                         # double that json stores round-trips it exactly.
                         "probability": float(probability)}
                        for probability, label in predictions
                    ],
                    "sentenceVector": packed.hex(),
                })

            target = os.path.join(GOLDEN, name + ".json")
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(
                    {"model": name, "loss": loss, "wordNgrams": word_ngrams,
                     "k": K, "cases": cases},
                    handle, ensure_ascii=False, indent=1)
                handle.write("\n")
            print(f"{path}: {os.path.getsize(path)} bytes, "
                  f"{len(cases)} cases -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
