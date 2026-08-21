import 'binary_reader.dart';

/// The loss function the model was trained with.
enum FastTextLoss {
  /// Labels sit at the leaves of a Huffman tree, so scoring one costs a walk
  /// down the tree instead of a pass over every label. What `lid.176` uses.
  hierarchicalSoftmax,

  /// Trained against a sample of wrong labels; at inference each label gets
  /// an independent sigmoid.
  negativeSampling,

  /// One distribution over all labels.
  softmax,

  /// An independent yes/no decision per label, for multi-label models.
  oneVsAll;

  static FastTextLoss _fromCode(int code) => switch (code) {
    1 => .hierarchicalSoftmax,
    2 => .negativeSampling,
    3 => .softmax,
    4 => .oneVsAll,
    _ => throw FormatException('unknown loss code: $code'),
  };
}

/// The model architecture.
enum FastTextArchitecture {
  /// Word vectors trained to predict a word from its context.
  cbow,

  /// Word vectors trained to predict the context from a word.
  skipgram,

  /// A classifier with labels — the only kind that can predict anything, and
  /// the only kind this package accepts.
  supervised;

  static FastTextArchitecture _fromCode(int code) => switch (code) {
    1 => .cbow,
    2 => .skipgram,
    3 => .supervised,
    _ => throw FormatException('unknown architecture code: $code'),
  };
}

/// The hyper-parameters stored in the model header.
///
/// For `lid.176.ftz`: `dim=16 minn=2 maxn=4 bucket=2000000 wordNgrams=1`,
/// hierarchical softmax loss, supervised architecture.
class ModelArgs {
  const ModelArgs._({
    required this.dim,
    required this.contextWindow,
    required this.epoch,
    required this.minCount,
    required this.negatives,
    required this.wordNgrams,
    required this.loss,
    required this.architecture,
    required this.bucket,
    required this.minCharNgram,
    required this.maxCharNgram,
    required this.lrUpdateRate,
    required this.samplingThreshold,
  });

  /// Reads the header from [reader].
  ///
  /// Internal: [BinaryReader] is not part of the public API, so there is no
  /// way to call this from outside the package.
  factory ModelArgs.read(BinaryReader reader, int formatVersion) {
    final dim = reader.int32();
    final contextWindow = reader.int32();
    final epoch = reader.int32();
    final minCount = reader.int32();
    final negatives = reader.int32();
    final wordNgrams = reader.int32();
    final loss = FastTextLoss._fromCode(reader.int32());
    final architecture = FastTextArchitecture._fromCode(reader.int32());
    final bucket = reader.int32();
    final minCharNgram = reader.int32();
    var maxCharNgram = reader.int32();
    final lrUpdateRate = reader.int32();
    final samplingThreshold = reader.float64();

    // Backwards compatibility: format 11 supervised models did not use
    // character n-grams, even though the file stores a non-zero value.
    if (formatVersion == 11 &&
        architecture == FastTextArchitecture.supervised) {
      maxCharNgram = 0;
    }

    // Both kinds of n-gram live in the same hash table, and fastText zeroes
    // its size itself when it trains a model that needs none — so an empty
    // table is ordinary, and an empty table beside n-grams to put in it is a
    // file that contradicts itself. Without this the n-grams would simply be
    // dropped and the model would answer from whatever was left.
    if (bucket < 0 || (bucket == 0 && (maxCharNgram > 0 || wordNgrams > 1))) {
      throw FormatException(
        'the model has an n-gram table of $bucket entries, but asks for '
        'character n-grams up to $maxCharNgram and word n-grams up to '
        '$wordNgrams',
      );
    }

    return ModelArgs._(
      dim: dim,
      contextWindow: contextWindow,
      epoch: epoch,
      minCount: minCount,
      negatives: negatives,
      wordNgrams: wordNgrams,
      loss: loss,
      architecture: architecture,
      bucket: bucket,
      minCharNgram: minCharNgram,
      maxCharNgram: maxCharNgram,
      lrUpdateRate: lrUpdateRate,
      samplingThreshold: samplingThreshold,
    );
  }

  /// Vector dimension.
  final int dim;

  /// Size of the context window used during training.
  final int contextWindow;

  /// How many passes over the training corpus were made.
  final int epoch;

  /// Words rarer than this were dropped from the dictionary. `lid.176` used
  /// 1000, which is why its vocabulary is small and it leans on character
  /// n-grams.
  final int minCount;

  /// Number of negative samples per update, used by
  /// [FastTextLoss.negativeSampling].
  final int negatives;

  /// Word n-gram length; 1 means single words only.
  final int wordNgrams;

  /// The loss the model was trained with; it decides the shape of the
  /// output layer.
  final FastTextLoss loss;

  /// What kind of model this is. Only [FastTextArchitecture.supervised] can
  /// predict labels.
  final FastTextArchitecture architecture;

  /// Size of the character n-gram hash table.
  final int bucket;

  /// Minimum and maximum character n-gram length.
  final int minCharNgram;
  final int maxCharNgram;

  /// How often the learning rate was updated during training.
  final int lrUpdateRate;

  /// The subsampling threshold for frequent words during training.
  final double samplingThreshold;

  @override
  String toString() =>
      'ModelArgs(dim: $dim, charNgrams: '
      '$minCharNgram..$maxCharNgram, bucket: $bucket, '
      'wordNgrams: $wordNgrams, loss: ${loss.name}, '
      'architecture: ${architecture.name})';
}
