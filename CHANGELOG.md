## Unreleased

### Fixed

- Word n-grams are applied. `wordNgrams` was read from the header and
  published, but never used: a model trained with `-wordNgrams 2` silently
  lost every bigram row and answered from what was left.
- `d += a * b` in the dot product is no longer read as a fused multiply-add,
  which the reference build does not emit. It moved 27 of 41 cases of
  `lid.176.bin`, by up to 398 ulps.
- Three breaks in the float32 rounding: the softmax fed a double difference
  to `exp`, the sigmoid table left its `exp` in double while the identical
  expression beside it was rounded, and `threshold` meant different things on
  different losses.
- The k best labels are kept in fastText's heap rather than a sorted insert,
  so labels that score the same come back in the same order as the original.
- `identify` answers `null` for text that is only whitespace. It used to
  return English at 12.5%, because fastText appends a newline and scores the
  end-of-sentence token; `predict` still does, since it reproduces the
  original.
- A corrupt or truncated model fails as the documented `FormatException`
  rather than a `StateError` or a `RangeError` from inside the reader, and
  header counts are checked before they size an allocation.
- The header, the dictionary counts and the matrix shapes are validated
  against each other at load. A dimension the matrices do not have used to
  leave half the hidden vector at zero and answer confidently anyway.
- Scores that are not finite numbers are refused. A NaN passed every
  comparison as false, which made hierarchical softmax return the likeliest
  label last.
- A download that stops early is refused, including the chunked case with no
  `Content-Length`, where the file used to be renamed into place and cached.
- A file on disk that is not a model is fetched again instead of failing
  every call for good; `loadOrDownloadModel` takes `force`.
- The connection, the response and the gaps in the body are bounded by
  timeouts. A silent server used to hang the caller forever.
- `tool/download_model.dart` no longer reports success for an error page
  saved under the model's name.
- A dense model no longer keeps its file in memory beside the copy of its
  matrix: about 125 MB for `lid.176.bin`. Quantized models still keep it, on
  purpose.
- One enormous token no longer leaves the dictionary holding a buffer its
  size for good.

### Changed

- Parity is checked exactly rather than to nine decimal places within `1e-6`,
  and the suite carries five reference models of its own for the losses and
  the word n-grams `lid.176` cannot reach. Missing weights fail instead of
  skipping when `LANG_ID_REQUIRE_MODELS` or `CI` is set. The bit-exactness
  claim now names the one difference it cannot reproduce — a platform's
  `expf` is not always correctly rounded — and bounds it.

## 0.1.0

- First release.
- Reads fastText models: `.bin` (dense matrix) and `.ftz` (product quantization).
- Inference for supervised models with any loss: hierarchical softmax, plain
  softmax, negative sampling, one-vs-all.
- `sentenceVector` exposes the embedding behind a prediction.
- Bit-exact parity with the reference fastText implementation, checked for both
  predictions and sentence vectors on `lid.176.ftz` and `lid.176.bin`.
- `ModelDownloader` and `loadOrDownloadModel` fetch the weights into a
  directory of your choosing: the package carries no model of its own.
- No FFI and no native code: runs on the VM, in AOT builds and on the web.
