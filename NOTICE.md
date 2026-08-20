# Licences

## Package code

The `lang_id` code is distributed under the MIT licence (see `LICENSE`).
Model weights are **not** part of the package.

## fastText models

The package reads Facebook's language identification models, `lid.176.bin`
and `lid.176.ftz`. The model files themselves are not included in the
repository and are downloaded separately
(`dart run tool/download_model.dart`).

The models are distributed under the
[Creative Commons Attribution-Share-Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
licence, inherited from their training data (Wikipedia, Tatoeba, SETimes).
Source: <https://fasttext.cc/docs/en/language-identification.html>

What this means in practice:

* **Attribution.** When you distribute the model file, credit the author
  (Facebook, Inc.), link to the licence, and state whether you modified it.
* **ShareAlike.** If you *adapt* the model — convert it to your own format,
  requantize, prune or fine-tune it — the resulting file must be released
  under the same licence. The code that reads the model is not a derivative
  work and stays under your own licence: the CC BY-SA text says explicitly
  that including a work in a collection does not subject the collection to
  the terms of the licence.
* The obligations arise **on distribution**. Running inference on your own
  server, where only a language code leaves the machine, is not distribution
  of the model.
* You may not apply technological measures to the model file that restrict
  the recipient's access to it.

This is a reading of the licence text, not legal advice. For a commercial
release it is worth showing it to a lawyer.

## Model format

The binary format was worked out against the sources of
[facebookresearch/fastText](https://github.com/facebookresearch/fastText)
(MIT licence). No fastText code was copied into this package.
