"""What the golden generators ask of the original fastText, in one place.

Both `generate_golden.py` and `generate_fixtures.py` go around the Python
wrapper, which is incompatible with numpy 2, and talk to the pybind module
directly. That is a private interface, so it lives here once rather than in
two copies that can drift apart.
"""
import struct

import fasttext_pybind
import numpy as np


def predict(model, text, k):
    """The k best labels as (probability, label) pairs, exactly as fastText
    scores them. The binding appends \\n itself; this does what it does."""
    return model.f.predict(text + "\n", k, 0.0, "strict")


def sentence_vector_hex(model, text):
    """The sentence vector as raw little-endian float32, so that the
    comparison in the tests is bit-exact rather than rounded through
    decimal."""
    vector = fasttext_pybind.Vector(model.get_dimension())
    model.f.getSentenceVector(vector, text + "\n")
    return b"".join(
        struct.pack("<f", float(x)) for x in np.asarray(vector)
    ).hex()
