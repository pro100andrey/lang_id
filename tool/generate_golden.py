#!/usr/bin/env python3
"""Regenerates the reference predictions from the original fastText.

Run this whenever the model or the set of texts changes; the output lands in
test/golden, which the parity test checks against.

    python3 -m venv .venv && .venv/bin/pip install fasttext-wheel
    .venv/bin/python tool/generate_golden.py
"""
import json
import os
import struct
import sys

import fasttext
import fasttext_pybind

# Reference texts: the major writing systems, closely related languages
# (which the model tells apart worst of all) and degenerate inputs.
TEXTS = [
    "Привет, как дела? Сегодня отличная погода.",
    "The quick brown fox jumps over the lazy dog",
    "Bonjour tout le monde, comment allez-vous ?",
    "Die Katze sitzt auf der Matte und schläft",
    "El rápido zorro marrón salta sobre el perro perezoso",
    "Il gatto si siede sul tappeto e dorme tranquillo",
    "O rápido cão castanho salta sobre o cão preguiçoso",
    "Wie gaat het met jou vandaag mijn vriend",
    "Szybki brązowy lis przeskakuje nad leniwym psem",
    "Мова програмування Dart створена компанією Google",
    "Беларуская мова належыць да ўсходнеславянскай групы",
    "Българският език е индоевропейски език от славянската група",
    "Rychlá hnědá liška skáče přes líného psa",
    "Den snabba bruna räven hoppar över den lata hunden",
    "Nopea ruskea kettu hyppää laiskan koiran yli",
    "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar",
    "اللغة العربية من أقدم اللغات السامية",
    "השפה העברית היא שפה שמית עתיקה",
    "زبان فارسی یکی از زبان‌های هندواروپایی است",
    "हिन्दी भारत की राजभाषा है",
    "ภาษาไทยเป็นภาษาราชการของประเทศไทย",
    "今日はいい天気ですね",
    "中文是世界上使用人数最多的语言",
    "한국어는 한반도에서 사용되는 언어입니다",
    "Tiếng Việt là ngôn ngữ chính thức của Việt Nam",
    "Η ελληνική γλώσσα έχει μακρά ιστορία",
    "ქართული ენა არის საქართველოს ოფიციალური ენა",
    "Հայերենը հնդեվրոպական լեզու է",
    "Bahasa Indonesia adalah bahasa resmi Republik Indonesia",
    "Kiswahili ni lugha inayozungumzwa Afrika Mashariki",
    # degenerate inputs
    "",
    " ",
    "   \t  ",
    "12345 67890",
    "!!! ??? ...",
    "😀 🎉 🚀",
    "OK",
    "a",
    "Hello мир 世界",
    "www.example.com/path?query=1",
    "The quick brown fox " * 40,
]

K = 5


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "test", "golden")
    os.makedirs(out_dir, exist_ok=True)
    for name in ("lid.176.ftz", "lid.176.bin"):
        path = os.path.join(os.path.dirname(__file__), "..", "models", name)
        if not os.path.exists(path):
            print(f"skipping {name}: file not found", file=sys.stderr)
            continue
        model = fasttext.load_model(path)
        cases = []
        for text in TEXTS:
            # The Python binding appends \n itself; we go around the wrapper
            # because it is incompatible with numpy 2.
            predictions = model.f.predict(text + "\n", K, 0.0, "strict")
            # The sentence vector as raw little-endian float32, so the
            # comparison is bit-exact rather than rounded through decimal.
            vector = fasttext_pybind.Vector(model.get_dimension())
            model.f.getSentenceVector(vector, text + "\n")
            import numpy as np
            packed = b"".join(
                struct.pack("<f", float(x)) for x in np.asarray(vector)
            )

            cases.append({
                "text": text,
                "predictions": [
                    {"label": label.replace("__label__", ""),
                     "probability": round(float(probability), 9)}
                    for probability, label in predictions
                ],
                "sentenceVector": packed.hex(),
            })
        target = os.path.join(out_dir, name.replace(".", "_") + ".json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump({"model": name, "k": K, "cases": cases}, handle,
                      ensure_ascii=False, indent=1)
            handle.write("\n")
        print(f"{target}: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
