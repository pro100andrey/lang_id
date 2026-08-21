import 'dart:convert';
import 'dart:typed_data';

/// Builds a tiny but complete fastText model in memory.
///
/// Three words and two labels by default, dense matrices, plain softmax, no
/// character n-grams. `alpha` maps to label `x`, `beta` to label `y`. Small
/// enough to keep in a test, real enough to exercise the whole reader.
///
/// The named parameters exist so a test can ask for the shape it needs: a
/// different loss, word n-grams over a hash table, or labels that score
/// exactly the same. Everything else stays fixed, which is what makes
/// [syntheticHeaderOffsets] usable for corrupting a single field.
///
/// Layout: signature, args, dictionary, quantization flag, input matrix,
/// output quantization flag, output matrix.
Uint8List buildSyntheticModel({
  int loss = lossSoftmax,
  int labelCount = 2,
  int wordNgrams = 1,
  int bucket = 0,
  bool tiedLabels = false,
}) {
  const dim = 2;
  final words = [
    ('</s>', 3, 0),
    ('alpha', 2, 0),
    ('beta', 1, 0),
    for (var i = 0; i < labelCount; i++)
      ('__label__${_labelNames[i]}', 10 - i, 1),
  ];
  final wordCount = words.length - labelCount;
  final inputRows = wordCount + bucket;

  final out = ByteSink()
    ..int32(793712314) // signature
    ..int32(12) // format version
    ..int32(dim)
    ..int32(5) // ws
    ..int32(1) // epoch
    ..int32(1) // minCount
    ..int32(0) // neg
    ..int32(wordNgrams)
    ..int32(loss)
    ..int32(3) // model = supervised
    ..int32(bucket)
    ..int32(0) // minn
    ..int32(0) // maxn: no character n-grams
    ..int32(100) // lrUpdateRate
    ..float64(1e-4) // t
    ..int32(words.length) // size
    ..int32(wordCount) // nwords
    ..int32(labelCount) // nlabels
    ..int64(6) // ntokens
    ..int64(-1); // pruneidx: the model is not pruned

  for (final (word, count, type) in words) {
    out
      ..cString(word)
      ..int64(count)
      ..uint8(type);
  }

  return (out
        ..uint8(0) // the input matrix is not quantized
        ..int64(inputRows)
        ..int64(dim)
        ..float32s([
          0, 0, 1, 0, 0, 1, // </s>, alpha, beta
          // Hash-table rows, each one different, so a word n-gram landing in
          // a bucket actually moves the hidden vector.
          for (var i = 0; i < bucket; i++) ...[(i % 7) - 3, 3 - (i % 5)],
        ])
        ..uint8(0) // the output matrix is not quantized
        ..int64(labelCount)
        ..int64(dim)
        ..float32s([
          for (var i = 0; i < labelCount; i++)
            if (tiedLabels) ...const [0, 0] else if (i.isEven) ...const [
              10,
              0,
            ] else ...const [0, 10],
        ]))
      .bytes;
}

/// The UTF-8 bytes of [value], for hashing words the way the dictionary
/// does.
Uint8List utf8Bytes(String value) => Uint8List.fromList(utf8.encode(value));

/// Loss codes as the model file stores them.
const lossHierarchicalSoftmax = 1;
const lossNegativeSampling = 2;
const lossSoftmax = 3;
const lossOneVsAll = 4;

const _labelNames = ['x', 'y', 'z', 'w', 'v', 'u', 't', 's'];

/// Byte offsets of the header fields inside [buildSyntheticModel].
///
/// The header is fixed-width and comes before anything variable, so a test
/// can rewrite one field with [patchInt32] and watch the reader reject the
/// file — which is the only way to reach the checks that guard against a
/// corrupt model.
const ({
  int bucket,
  int dim,
  int labelCount,
  int loss,
  int pruneIndexSize,
  int size,
  int wordCount,
  int wordNgrams,
})
syntheticHeaderOffsets = (
  dim: 8,
  wordNgrams: 28,
  loss: 32,
  bucket: 40,
  size: 64,
  wordCount: 68,
  labelCount: 72,
  pruneIndexSize: 84,
);

/// A copy of [model] with the int32 at [offset] replaced by [value].
Uint8List patchInt32(Uint8List model, int offset, int value) {
  final copy = Uint8List.fromList(model);
  ByteData.view(copy.buffer).setInt32(offset, value, Endian.little);

  return copy;
}

/// A little-endian byte writer, enough to lay out a model file by hand.
class ByteSink {
  final _builder = BytesBuilder();

  Uint8List get bytes => _builder.toBytes();

  void uint8(int value) => _builder.addByte(value);

  void int32(int value) => _add(4, (d) => d.setInt32(0, value, Endian.little));

  void int64(int value) => _add(8, (d) {
    d
      ..setUint32(0, value & 0xFFFFFFFF, Endian.little)
      ..setInt32(4, value < 0 ? -1 : value ~/ 0x100000000, Endian.little);
  });

  void float64(double value) =>
      _add(8, (d) => d.setFloat64(0, value, Endian.little));

  void float32s(List<num> values) => _builder.add(
    Uint8List.view(
      Float32List.fromList([for (final v in values) v.toDouble()]).buffer,
    ),
  );

  void cString(String value) {
    _builder
      ..add(utf8.encode(value))
      ..addByte(0);
  }

  void _add(int size, void Function(ByteData) write) {
    final data = ByteData(size);
    write(data);
    _builder.add(data.buffer.asUint8List());
  }
}
