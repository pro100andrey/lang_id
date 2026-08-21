# lang_id

Language identification in pure Dart. The package reads
[fastText](https://fasttext.cc) models itself — both plain `.bin` and
quantized `.ftz` — and computes the prediction itself. No FFI, no native
libraries, no C++ build step: it runs on the VM, in AOT builds and on the web.

```sh
dart pub add lang_id
```

```dart
import 'package:lang_id/lang_id_io.dart';

// Downloads the model on the first run, reads it from disk afterwards.
final identifier = await loadOrDownloadModel(
  PretrainedModel.compact,
  directory: 'models',
);

print(identifier.identify('Привіт, як справи? Сьогодні чудова погода.'));
// uk 96.7%
print(identifier.predict('Bonjour', k: 3));
// [fr 90.2%, en 5.5%, de 0.7%]
```

## The model

The weights are **not** shipped with the package — they are published under a
different licence and the full model is 125 MB. The package fetches them for
you, into a directory you choose.

```dart
final downloader = ModelDownloader();
try {
  final file = await downloader.download(
    .compact, // lid.176.ftz, 916 KB
    directory: 'assets/models',
    onProgress: (p) => stdout.write('\r$p'),
  );
  print(file.path); // assets/models/lid.176.ftz
} finally {
  await downloader.close();
}
```

* A model already in the directory is returned as is; pass `force: true` to
  fetch it again.
* The download goes to a `.part` file and is renamed into place only once it
  is complete and its header checks out, so an interrupted run never leaves a
  half-written model behind.
* `fileIn(directory, model)` tells you where the file will land without
  touching the network — handy for deciding whether to show a progress bar.
* `ModelDownloader(baseUrl: ...)` points at a mirror or an internal copy;
  `ModelDownloader(httpClient: ...)` reuses a client you already configured
  with proxies, certificates or timeouts.
* Failures come back as `ModelDownloadException`, including the case where
  what arrived is not a fastText model at all — a captive portal or an error
  page, for instance. A file already on disk that turns out not to be a model
  is thrown away and fetched again, rather than reported forever.
* A body that stops early is refused: against the announced `Content-Length`
  when there is one, and by reading the file as a model when there is not,
  since a chunked response cut in half ends just as quietly as a whole one.
* Reaching the server, getting an answer and the gaps in between are all
  bounded — `ModelDownloader(connectionTimeout: ..., stallTimeout: ...)` —
  so a mirror that accepts the connection and says nothing cannot hang the
  app that is starting up.

The same thing from the command line:

```sh
dart run tool/download_model.dart                 # lid.176.ftz into models/
dart run tool/download_model.dart --full          # lid.176.bin, 125 MB
dart run tool/download_model.dart --out assets    # somewhere else
```

Both models tell 176 languages apart. `.ftz` is compressed with product
quantization and gives up almost nothing on typical text; start with it.

| | `lid.176.ftz` | `lid.176.bin` |
| --- | --- | --- |
| File size | 916 KB | 125 MB |
| Memory for the model | ~1 MB | ~125 MB |
| Load time | ~20 ms | ~50 ms |
| Prediction | ~5 µs | ~5 µs |

Measured on Apple Silicon, Dart 3.13, short strings, a single isolate — a bit
under 200 thousand predictions per second on one thread. Quantization costs
nothing at prediction time: decoding a row is eight table lookups, and the
work is dominated by hashing the character n-grams either way.

## Usage

```dart
import 'dart:io';
import 'package:lang_id/lang_id.dart';

// The core of the package does not depend on dart:io, so the bytes can come
// from anywhere: Flutter assets, the network, memory. Use lang_id_io.dart for
// loadModel / loadOrDownloadModel when you do have a filesystem.
final identifier = LanguageIdentifier.fromBytes(
    File('models/lid.176.ftz').readAsBytesSync());

// The most likely language, or null when there is nothing to predict from.
final best = identifier.identify('Мова програмування Dart');
print('${best?.label} ${best?.probability}');   // uk 0.9649…

// Several candidates, cut off by probability. `k: -1` asks for all of them.
const text = 'Мова програмування Dart створена компанією Google';
for (final p in identifier.predict(text, k: 5, threshold: 0.01)) {
  print('$p');
}

print(identifier.languages.length);   // 176
print(identifier.info);               // dim, loss, dictionary size

// The embedding behind the prediction, if you want it on its own.
final vector = identifier.sentenceVector('Мова програмування Dart');
print(vector.length);                 // 16
```

A ready-made example lives in `example/lang_id_example.dart`. It needs no
setup step — the model is fetched on the first run:

```sh
dart run example/lang_id_example.dart                       # built-in demo
dart run example/lang_id_example.dart "Привіт, як справи?"  # your own text
cat file.txt | dart run example/lang_id_example.dart -      # line by line
```

Standard input is read only when asked for with `-`, so running the file with
no arguments always finishes instead of waiting for input that may never
arrive.

### Things worth knowing

* **Tokenization.** fastText splits a line on whitespace only; punctuation
  stays attached to the word. The package behaves the same way, so there is no
  need to clean the text up front — and if you want to match the original,
  you should not.
* **Multi-line text.** By default `joinLines: true`: newlines count as
  ordinary separators and the whole text is used. With `joinLines: false` the
  parse stops at the first newline, the way the fastText CLI behaves.
* **Short strings.** The model was trained on sentences; on two or three words
  it errs noticeably more often, especially between closely related languages.
  Watch the probability and set a `threshold`.
* **An instance is not thread-safe** — it reuses an internal buffer. Keep one
  per isolate.
* **Probabilities can nudge past 1**, to around `1.0001` on a certain answer,
  and `predict` usually returns fewer than `k` entries. Both come from the
  same place: fastText adds `1e-5` under every logarithm so a zero
  probability stays finite, and it prunes candidates that fall below that
  floor. The numbers are reproduced as the original computes them rather than
  tidied up, because tidying them would break the parity below.
* **Loading the 125 MB model blocks** for tens of milliseconds and allocates
  125 MB. Handing the loaded identifier to another isolate copies its data,
  which defeats the point at that size — load and predict inside one
  long-lived isolate instead. The compact model loads in about 20 ms, so it
  rarely needs any of this.

## In a Flutter app

Put the model in a directory the app owns and let the package fetch it:

```dart
import 'package:path_provider/path_provider.dart';

final directory = await getApplicationSupportDirectory();
final identifier = await loadOrDownloadModel(
  PretrainedModel.compact,
  directory: directory.path,
  onProgress: (p) => setState(() => _progress = p.fraction),
);
```

The alternative is bundling the 916 KB `.ftz` as an asset, which trades app
size for working offline on the very first run:

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/lid.176.ftz
```

```dart
final data = await rootBundle.load('assets/lid.176.ftz');
// The offset and the length matter: a ByteData is a window onto a buffer
// that may hold other assets too.
final identifier = LanguageIdentifier.fromBytes(
    data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
```

Note that bundling means you are **distributing** the weights, so the
CC BY-SA attribution applies to your app — see `NOTICE.md`. Fetching them at
run time does not.

## Parity with the original

The answers match the reference fastText implementation **bit for bit**, and
the tests say so in those terms: labels, probabilities and sentence vectors
are all compared exactly, the vectors and the probabilities as float32 values
rather than through decimal.

`test/parity_test.dart` runs 41 cases against each of the two `lid.176`
models: thirty writing systems, closely related languages, and degenerate
inputs such as the empty string, emoji and bare punctuation. `lid.176` is one
model with one loss and no word n-grams, so `test/fixture_parity_test.dart`
adds five more, trained by the original fastText and small enough to live in
the repository: plain softmax, negative sampling, one-vs-all, and two with
`wordNgrams: 2` over a corpus whose labels are separated by word order alone.
All the reference answers live in `test/golden`, regenerated by
`tool/generate_golden.py` and `tool/generate_fixtures.py`.

Matching it means reproducing not just the algorithm but the arithmetic.
fastText runs inference in `float`, and in the tail of the output, where
probabilities bottom out at `1e-5`, Dart's extra precision swaps neighbouring
labels around. Every intermediate is rounded exactly where C++ rounds it. Two
spots are worth naming:

* The sigmoid sums `1 + exp(-x)` in float32, so for large `x` the sum
  collapses to exactly one and `1 - f` to zero. In double it yields one ulp
  instead of zero, and the logarithm of a rare branch drifts by several
  digits. The lookup table beside it is written against a double literal and
  does not — the two look identical and are not.
* `d += a * b` in the dot product is **not** read as a fused multiply-add.
  C++ is allowed to contract it into one, and the build these goldens come
  from does not, so the product is rounded on its own. Reading it the other
  way moves 27 of the 41 dense cases, by up to 398 ulps.

One difference is not reproducible, and the tests bound it rather than hide
it. fastText takes its exponentials in `float`, and a platform's `expf` is
not always the correctly rounded result — Dart's `exp` rounded to float32 is.
Where they disagree it is by one ulp, which the walk down the Huffman tree
can grow to a few by the time it reaches a probability: on the machine these
goldens were generated on, three of the 41 quantized cases differ, by at most
eight ulps. The suite allows that much and no more, which is far below what
any mistake in the arithmetic costs.

## What is supported

* fastText model formats 11 and 12, little-endian.
* Dense (`.bin`) and product-quantized (`.ftz`) matrices.
* The supervised architecture with any loss: hierarchical softmax (what
  `lid.176` uses), plain softmax, negative sampling, one-vs-all.
* Character n-grams and word n-grams (`-wordNgrams`), pruned dictionaries,
  quantized row norms.

Word-vector models (`cbow`, `skipgram`) are rejected: they have no labels, so
there is nothing to predict.

## The web

`package:lang_id/lang_id.dart` does not depend on `dart:io`. Everything that
would trip up dart2js is worked around: 64-bit fields are assembled from two
32-bit words, and the n-gram hash multiplies by halves, because the product of
two 32-bit numbers does not fit in a double's mantissa. The `binary_reader`,
`dictionary` and `synthetic_model` tests run in Chrome:

```sh
dart test -p chrome test/binary_reader_test.dart test/dictionary_test.dart \
    test/synthetic_model_test.dart
```

Loading weights from disk lives in a separate library,
`package:lang_id/lang_id_io.dart`, which does require `dart:io`.

## How it works

| File | What is inside |
| --- | --- |
| `lib/src/binary_reader.dart` | cursor over the file, 64-bit fields without `getInt64` |
| `lib/src/model_args.dart` | the header of hyper-parameters |
| `lib/src/dictionary.dart` | FNV-1a hash, character n-grams, tokenization |
| `lib/src/matrix.dart` | dense and quantized matrices, product quantizer |
| `lib/src/output_layer.dart` | Huffman tree, softmax, sigmoids |
| `lib/src/float32.dart` | rounding intermediate results to float32 |
| `lib/src/model_format.dart` | format signature and version constants |
| `lib/src/language_identifier.dart` | assembling the model and predicting |
| `lib/src/model_downloader.dart` | fetching the weights into a directory |

The binary format was worked out against the sources of
[facebookresearch/fastText](https://github.com/facebookresearch/fastText);
no code was copied from there.

## Development

```sh
dart pub get
dart run tool/download_model.dart          # lid.176.ftz
dart run tool/download_model.dart --full   # lid.176.bin, for the whole suite
dart test
dart analyze
```

Tests that need the weights are reported as skipped when the files are
missing, which is right for a working copy and wrong for anything automated:
set `LANG_ID_REQUIRE_MODELS=1` (as CI does) and a missing model fails the run
instead. The reference models in `test/fixtures` are a few kilobytes and are
committed, so they always run. The downloader is tested against a local
`HttpServer`, so the suite stays offline and deterministic.

Regenerating the reference answers needs the original fastText:

```sh
python3 -m venv .venv && .venv/bin/pip install fasttext-wheel
.venv/bin/python tool/generate_golden.py      # lid.176, needs the weights
.venv/bin/python tool/generate_fixtures.py    # the small trained models
```

Test data deliberately contains text in many languages, Russian among them —
that is the input a language identifier is supposed to be checked against.

## Licences

Code — MIT (`LICENSE`). **The fastText models are CC BY-SA 3.0**; they are not
committed to the repository and are downloaded separately. If you ship the
weights inside your product or adapt them, read `NOTICE.md`: it spells out
what requires attribution and what requires share-alike.
