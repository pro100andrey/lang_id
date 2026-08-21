import 'dart:math' as math;
import 'dart:typed_data';

import 'binary_reader.dart';
import 'dictionary.dart';
import 'float32.dart';
import 'matrix.dart';
import 'model_args.dart';
import 'model_format.dart';
import 'output_layer.dart';
import 'prediction.dart';

/// Facts about a loaded model.
class ModelInfo {
  /// Internal: built by [LanguageIdentifier.fromBytes] while reading a file.
  const ModelInfo({
    required this.formatVersion,
    required this.args,
    required this.quantized,
    required this.wordCount,
    required this.labelCount,
    required this.tokenCount,
  });

  /// File format version; the current one is 12.
  final int formatVersion;

  /// The hyper-parameters the model was trained with.
  final ModelArgs args;

  /// `true` for `.ftz` models compressed with product quantization.
  final bool quantized;

  /// Words in the dictionary. Small for `lid.176`, which relies on
  /// character n-grams rather than whole words.
  final int wordCount;

  /// Labels the model can predict — languages, for `lid.176`.
  final int labelCount;

  /// Number of tokens in the corpus the model was trained on.
  final int tokenCount;

  @override
  String toString() =>
      'ModelInfo(version: $formatVersion, '
      'quantized: $quantized, words: $wordCount, labels: $labelCount, '
      'dim: ${args.dim}, loss: ${args.loss.name})';
}

/// A language identifier backed by a fastText model.
///
/// Reads both `.bin` and quantized `.ftz`; the matrix representation is
/// chosen from a flag in the file and never shows up in the API. Strictly
/// speaking this reads any supervised fastText model — labels come back as
/// [Prediction.label], and for `lid.176` those labels are language codes.
///
/// ```dart
/// final id = LanguageIdentifier.fromBytes(bytes);
/// print(id.identify('Привіт, як справи?')); // uk 84.4%
/// ```
///
/// An instance reuses an internal buffer and is therefore not thread-safe;
/// keep one per isolate.
class LanguageIdentifier {
  LanguageIdentifier._(
    this._dictionary,
    this._input,
    this._output,
    this._labels,
    this.info,
  ) : _hidden = Float32List(info.args.dim);

  /// Parses a model from the bytes of a `.bin` or `.ftz` file.
  ///
  /// A quantized model keeps the bytes: its matrix codes are read as windows
  /// over the source buffer rather than copied, which is most of why it is
  /// small. A dense one does not — its matrix is copied out and the buffer
  /// can be collected as soon as you let go of it.
  factory LanguageIdentifier.fromBytes(Uint8List bytes) {
    final reader = BinaryReader(bytes);
    final magic = reader.int32();
    final formatVersion = reader.int32();
    checkFastTextHeader(magic, formatVersion);

    final args = ModelArgs.read(reader, formatVersion);
    if (args.architecture != FastTextArchitecture.supervised) {
      throw FormatException(
        'a supervised model is required, but this is '
        '${args.architecture.name}: it has no labels to predict',
      );
    }

    final dictionary = Dictionary.read(reader, args);

    final quantized = reader.boolean();
    final input = Matrix.read(reader, quantized: quantized);
    if (!quantized && dictionary.isPruned) {
      throw const FormatException(
        'pruned dictionary in an unquantized '
        'model: the file is corrupt or was built by an old fastText',
      );
    }

    final quantizedOutput = reader.boolean();
    final output = Matrix.read(reader, quantized: quantized && quantizedOutput);
    _validateShapes(args, dictionary, input, output);

    return LanguageIdentifier._(
      dictionary,
      input,
      OutputLayer.create(args, output, dictionary.labelCounts),
      List.unmodifiable(dictionary.labelNames),
      ModelInfo(
        formatVersion: formatVersion,
        args: args,
        quantized: quantized,
        wordCount: dictionary.wordCount,
        labelCount: dictionary.labelCount,
        tokenCount: dictionary.tokenCount,
      ),
    );
  }

  /// Checks that the matrices match the header and the dictionary.
  ///
  /// Nothing in the file says these have to agree, and when they do not the
  /// interesting case is the quiet one: a header claiming more columns than
  /// the input matrix has leaves the tail of the hidden vector at zero, and
  /// the model answers with a confident label computed from half a vector.
  /// The rest surface as a range error from somewhere inside the arithmetic.
  /// Both are better as a [FormatException] at load time.
  static void _validateShapes(
    ModelArgs args,
    Dictionary dictionary,
    Matrix input,
    Matrix output,
  ) {
    // The count that every consumer below uses, rather than the header field
    // it is read from; Dictionary.read has already held the two to each
    // other.
    final labelCount = dictionary.size - dictionary.wordCount;
    if (labelCount == 0) {
      throw const FormatException('the model has no labels');
    }

    if (input.columns != args.dim || output.columns != args.dim) {
      throw FormatException(
        'the model says its vectors are ${args.dim} wide, but the input '
        'matrix has ${input.columns} columns and the output matrix '
        '${output.columns}',
      );
    }

    if (input.rows < dictionary.wordCount) {
      throw FormatException(
        'the input matrix has ${input.rows} rows, too few for '
        '${dictionary.wordCount} words',
      );
    }

    // Hierarchical softmax scores the internal nodes of the Huffman tree,
    // one row short of the labels; every other loss scores the labels
    // themselves.
    final rowsNeeded = args.loss == FastTextLoss.hierarchicalSoftmax
        ? labelCount - 1
        : labelCount;
    if (output.rows < rowsNeeded) {
      throw FormatException(
        'the output matrix has ${output.rows} rows, too few for $labelCount '
        'labels with ${args.loss.name} loss',
      );
    }
  }

  final Dictionary _dictionary;
  final Matrix _input;
  final OutputLayer _output;
  final List<String> _labels;
  final Float32List _hidden;

  /// What `k` means "as many as there are", the way fastText spells it.
  static const _everyLabel = -1;

  /// What the file said about itself: format version, dimensions, loss,
  /// dictionary sizes.
  final ModelInfo info;

  /// The labels the model tells apart, with the `__label__` prefix stripped.
  ///
  /// Language codes for `lid.176` — 176 of them — and whatever you trained
  /// on for a classifier of your own.
  ///
  /// Ordered by how often the label occurred in the training corpus, so the
  /// first entries are the best-represented ones.
  List<String> get labels => _labels;

  /// The most likely language of the text, or `null` when there is nothing
  /// to predict from: the text holds nothing but whitespace, or every one of
  /// its n-grams was dropped when the model was pruned.
  ///
  /// The empty case needs saying out loud, because fastText does not treat
  /// it as one. It appends a newline before parsing, which always yields the
  /// end-of-sentence token, which is always in the dictionary — so an empty
  /// line has something to score, and comes back as English at 12.5%. That
  /// is what [predict] returns, because it reproduces the original. This
  /// answers `null` instead, so that the obvious check on a form field means
  /// what it looks like.
  Prediction? identify(
    String text, {
    double threshold = 0.0,
    bool joinLines = true,
  }) {
    if (Dictionary.isBlank(text)) {
      return null;
    }

    final best = predict(text, threshold: threshold, joinLines: joinLines);

    return best.isEmpty ? null : best.first;
  }

  /// The [k] most likely languages, most likely first.
  ///
  /// [threshold] drops candidates below the given probability. [joinLines]
  /// folds multi-line text into a single line; with `false` the parse stops
  /// at the first newline, the way the fastText CLI behaves.
  ///
  /// `k: -1` asks for every language, which is what the number means to
  /// fastText itself. Expect far fewer than [labels] back even then, for the
  /// reason below.
  ///
  /// Expect fewer than [k] entries: fastText prunes candidates below its own
  /// floor of `1e-5` while searching, so asking for all 176 languages still
  /// returns a handful. See [Prediction.probability] for why the numbers can
  /// nudge just past 1.
  ///
  /// Throws [FormatException] when the weights turn out not to be finite
  /// numbers. That is a broken model file rather than anything about the
  /// text, but it can only be noticed here, while scoring.
  List<Prediction> predict(
    String text, {
    int k = 1,
    double threshold = 0.0,
    bool joinLines = true,
  }) {
    if (k < 1 && k != _everyLabel) {
      throw RangeError.value(k, 'k', 'must be at least 1, or -1 for all');
    }

    final wanted = k == _everyLabel ? _labels.length : k;
    final indices = _dictionary.lineToIndices(text, joinLines: joinLines);
    if (indices.isEmpty) {
      return const [];
    }
    _fillHidden(_hidden, indices);

    return [
      for (final scored in _output.predict(_hidden, wanted, threshold))
        Prediction(_labels[scored.label], float32(math.exp(scored.score))),
    ];
  }

  /// The sentence vector for [text] — what fastText calls
  /// `get_sentence_vector`, and exactly what [predict] feeds to the output
  /// layer.
  ///
  /// It is the mean of the vectors of every word and character n-gram in the
  /// line, so it is a text embedding you can use on its own: nearest
  /// neighbours, clustering, or as a feature for another model. Note that the
  /// language identification model was trained to separate languages, not
  /// meanings — two English sentences about different things still land close
  /// together.
  ///
  /// Returns a fresh list of `info.args.dim` values.
  ///
  /// There is no such thing as a text with nothing to embed, whatever this
  /// used to say: fastText appends a newline before parsing, which yields
  /// the end-of-sentence token, which is in every dictionary — so an empty
  /// string comes back as that token's own vector rather than as zeros. Use
  /// [Dictionary.isBlank], which is what [identify] does, to tell whether
  /// there was anything to look at.
  Float32List sentenceVector(String text, {bool joinLines = true}) {
    final vector = Float32List(info.args.dim);
    final indices = _dictionary.lineToIndices(text, joinLines: joinLines);
    if (indices.isNotEmpty) {
      _fillHidden(vector, indices);
    }
    return vector;
  }

  /// The hidden layer: the mean of the rows named by [indices].
  void _fillHidden(Float32List target, List<int> indices) {
    final dim = info.args.dim;
    target.fillRange(0, dim, 0);
    for (final index in indices) {
      _input.addRowTo(target, index);
    }
    // C++ passes the factor as a float, so round before applying it.
    final scale = float32(1.0 / indices.length);
    for (var i = 0; i < dim; i++) {
      target[i] *= scale;
    }
  }
}
