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
    this._languages,
    this.info,
  ) : _hidden = Float32List(info.args.dim);

  /// Parses a model from the bytes of a `.bin` or `.ftz` file.
  ///
  /// The bytes are retained: quantized matrix codes are read as windows over
  /// the source buffer, with no copying.
  factory LanguageIdentifier.fromBytes(Uint8List bytes) {
    final reader = BinaryReader(bytes);
    final magic = reader.int32();
    if (magic != fastTextMagic) {
      throw const FormatException('not a fastText model: wrong file signature');
    }

    final formatVersion = reader.int32();
    if (formatVersion > fastTextSupportedVersion) {
      throw FormatException(
        'format version $formatVersion is newer than '
        'the supported one ($fastTextSupportedVersion)',
      );
    }

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

    if (dictionary.labelCount == 0) {
      throw const FormatException('the model has no labels');
    }

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

  final Dictionary _dictionary;
  final Matrix _input;
  final OutputLayer _output;
  final List<String> _languages;
  final Float32List _hidden;

  /// What the file said about itself: format version, dimensions, loss,
  /// dictionary sizes.
  final ModelInfo info;

  /// The languages the model can tell apart: 176 codes for `lid.176`.
  ///
  /// Ordered by how often the label occurred in the training corpus, so the
  /// first entries are the best-represented languages.
  List<String> get languages => _languages;

  /// The most likely language of the text, or `null` when there is nothing
  /// to predict from: the text is empty, or every one of its n-grams was
  /// dropped when the model was pruned.
  Prediction? identify(
    String text, {
    double threshold = 0.0,
    bool joinLines = true,
  }) {
    final best = predict(text, threshold: threshold, joinLines: joinLines);
    return best.isEmpty ? null : best.first;
  }

  /// The [k] most likely languages, most likely first.
  ///
  /// [threshold] drops candidates below the given probability. [joinLines]
  /// folds multi-line text into a single line; with `false` the parse stops
  /// at the first newline, the way the fastText CLI behaves.
  ///
  /// Expect fewer than [k] entries: fastText prunes candidates below its own
  /// floor of `1e-5` while searching, so asking for all 176 languages still
  /// returns a handful. See [Prediction.probability] for why the numbers can
  /// nudge just past 1.
  List<Prediction> predict(
    String text, {
    int k = 1,
    double threshold = 0.0,
    bool joinLines = true,
  }) {
    if (k < 1) {
      throw RangeError.value(k, 'k', 'must be at least 1');
    }

    final indices = _dictionary.lineToIndices(text, joinLines: joinLines);
    if (indices.isEmpty) {
      return const [];
    }
    _fillHidden(_hidden, indices);

    return [
      for (final scored in _output.predict(_hidden, k, threshold))
        Prediction(_languages[scored.label], float32(math.exp(scored.score))),
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
  /// Returns a fresh list of `info.args.dim` values; a text with nothing to
  /// embed gives back zeros, as it does in fastText.
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
