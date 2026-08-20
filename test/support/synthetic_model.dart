import 'dart:convert';
import 'dart:typed_data';

/// Builds a tiny but complete fastText model in memory.
///
/// Two words and two labels, dense matrices, plain softmax, no character
/// n-grams. `alpha` maps to label `x`, `beta` to label `y`. Small enough to
/// keep in a test, real enough to exercise the whole reader.
///
/// Layout: signature, args, dictionary, quantization flag, input matrix,
/// output quantization flag, output matrix.
Uint8List buildSyntheticModel() {
  const dim = 2;
  const words = [
    ('</s>', 3, 0),
    ('alpha', 2, 0),
    ('beta', 1, 0),
    ('__label__x', 2, 1),
    ('__label__y', 1, 1),
  ];

  final out = ByteSink()
    ..int32(793712314) // signature
    ..int32(12) // format version
    ..int32(dim)
    ..int32(5) // ws
    ..int32(1) // epoch
    ..int32(1) // minCount
    ..int32(0) // neg
    ..int32(1) // wordNgrams
    ..int32(3) // loss = softmax
    ..int32(3) // model = supervised
    ..int32(0) // bucket
    ..int32(0) // minn
    ..int32(0) // maxn: no character n-grams
    ..int32(100) // lrUpdateRate
    ..float64(1e-4) // t
    ..int32(words.length) // size
    ..int32(3) // nwords
    ..int32(2) // nlabels
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
        ..int64(3) // rows: nwords + bucket
        ..int64(dim)
        ..float32s([0, 0, 1, 0, 0, 1]) // </s>, alpha, beta
        ..uint8(0) // the output matrix is not quantized
        ..int64(2) // rows: one per label
        ..int64(dim)
        ..float32s([10, 0, 0, 10])) // x, y
      .bytes;
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

  void float32s(List<double> values) =>
      _builder.add(Uint8List.view(Float32List.fromList(values).buffer));

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
