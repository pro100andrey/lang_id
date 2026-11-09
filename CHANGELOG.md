## Unreleased

### Fixed

- A write that fails during a download — a read-only directory, a full disk
  — is a `ModelDownloadException`. It used to surface as an unhandled error
  that killed the isolate. Each chunk is now written before the next is read,
  which also keeps a fast network from queueing the model in memory.
- A download that ends early for any reason — a timeout, a failed write,
  an exception from `onProgress` — aborts its request. A server that never
  answered, or a body nobody read, kept the socket and the process alive.
  The scratch file is opened before the request, so a directory that cannot
  be written fails without asking anything of the server.
- Scratch file names carry a random token. The process id and a per-isolate
  counter gave two isolates of one process the same file, and interleaved
  their downloads in it.
- `connectionTimeout` bounds reaching the server, and the wait for the
  response to begin counts as a stall, bounded by `stallTimeout`. A short
  connect timeout used to cut off a server that was only slow to answer.
- An error response whose body stalls is bounded by `stallTimeout` like any
  other body, and still reports its status.
- `directory: ''` means the working directory in `download` as it already
  did in `fileIn`, and a directory that cannot be created is a
  `ModelDownloadException` rather than a raw `FileSystemException`. Stale
  `.part` files are found however the directory is spelled.
- `loadOrDownloadModel` fetches again, once, when a cached file has a good
  header but does not read as a model — the truncated body 0.1.0 could leave
  behind. A file it has just downloaded and cannot read is reported, not
  fetched again.
- A `connectionTimeout` set on your own `HttpClient` is kept unless you pass
  one to `ModelDownloader`, and a `baseUrl` without a trailing slash keeps
  its last path segment.
- An input matrix too short for its n-gram hash table, or for the rows a
  prune index maps to, fails at load as a `FormatException` instead of as a
  `RangeError` on the first text that hashes high.
- A matrix size large enough to wrap the byte count on the VM fails as a
  `FormatException` instead of an `OutOfMemoryError`.
- `classify` answers `null` whenever the parse leaves nothing but the
  end-of-sentence token: a leading newline with `joinLines: false`, or only
  words the model cannot see.
- The order of labels with exactly equal scores is documented for what it
  is: libc++'s, which the goldens were generated against. A GCC build of
  fastText on Linux breaks such ties differently.
- `test/fasttext_classifier_test.dart` fails rather than skips when the
  weights are missing and `CI` or `LANG_ID_REQUIRE_MODELS` is set.

## 0.2.0

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
- `k: -1` asks for every label, the way it does in fastText, instead of being
  a range error.
- `Prediction.compareTo` sorts ascending, as `Comparable` asks, with the
  label breaking a tie so that it agrees with `==`. It used to sort
  descending, which quietly misled sorted sets and binary searches.
- A model asking for n-grams while declaring a hash table of no entries is
  refused instead of dropping them. An empty table on its own stays ordinary,
  since that is what fastText writes for a model trained without n-grams.
- Concurrent downloads of the same model no longer share one scratch file,
  interleave their chunks into it and rename the mixture into place.
- A compressed response no longer puts progress above 100%, and no longer
  looks like a truncated download when its announced length is compared with
  what unpacking produces.
- A mistake in the caller's own `onProgress` callback is no longer rewritten
  as a download failure with its stack trace thrown away.
- `fileIn` joins the file name onto the directory instead of resolving it as
  a URI, which normalized `..` by text — right only when nothing on the way
  is a symlink — and made an empty directory mean something other than the
  working directory.
- The reader and the downloader share one header check rather than a copy
  each, raising different exceptions for the same bytes.
- The example survives standard input that is not UTF-8, which is the input a
  language identifier is reached for, and reports a first run without the
  network instead of printing a stack trace.
- `tool/download_model.dart` no longer reads the argument after `--out`
  without looking at it: `--out --force` used to fetch into a directory named
  `--force`.
- The Flutter sample in the README keeps the offset and the length of the
  asset it loads, and the sample above it declares the variable it uses.
- `PretrainedModel.publishedSize` no longer claims to be a progress fallback
  it never was.

### Changed

- `LanguageIdentifier` is now `FastTextClassifier`, `identify` is `classify`
  and `languages` is `labels`, with no aliases left behind. The package reads any
  supervised fastText model, and a classifier trained on `__label__good` and
  `__label__garbage` has no languages to identify. The README shows how to
  read one.
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
