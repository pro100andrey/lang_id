import 'dart:typed_data';

import 'binary_reader.dart';
import 'float32.dart';

/// A matrix of model vectors.
///
/// Two implementations: [DenseMatrix] for `.bin` and [QuantizedMatrix] for
/// `.ftz`. Inference needs exactly two operations from a matrix — add a row
/// to a vector and dot a row with a vector — so quantization stays an
/// implementation detail and never leaks out.
abstract class Matrix {
  /// Reads a matrix, picking the representation from the model's
  /// quantization flag.
  factory Matrix.read(BinaryReader reader, {required bool quantized}) =>
      quantized ? QuantizedMatrix.read(reader) : DenseMatrix.read(reader);

  int get rows;
  int get columns;

  void addRowTo(Float32List target, int row);

  /// The dot product of [vector] with a row.
  ///
  /// Throws [FormatException] when the result is not a finite number, which
  /// means the weights are not either.
  double dotRow(Float32List vector, int row);
}

/// Refuses a score that is not a finite number.
///
/// fastText guards its own dot product against NaN, and the guard was worth
/// keeping. A NaN passes every comparison as false: the range checks of the
/// table-driven sigmoid let it through to a conversion that throws something
/// no caller could anticipate, and the two guards that prune the walk down
/// the Huffman tree stop firing, so the answer comes back in an order that
/// contradicts the "most likely first" the API promises.
///
/// Infinity is refused as well, which fastText does not do. It arrives one
/// step earlier — an infinite weight makes an infinite score, and the
/// softmax turns that into NaN when it subtracts the maximum — and there is
/// nothing a caller could do with the answer either way. Trained weights
/// never come near the size this needs.
double _requireNumber(double score) {
  if (!score.isFinite) {
    throw FormatException(
      'the model scored a text as $score: its weights contain NaN or an '
      'infinity',
    );
  }

  return score;
}

/// A dense float32 matrix, as found in unquantized `.bin` models.
class DenseMatrix implements Matrix {
  DenseMatrix._(this.rows, this.columns, this._data);

  factory DenseMatrix.read(BinaryReader reader) {
    final rows = reader.int64();
    final columns = reader.int64();
    if (rows < 0 || columns < 0) {
      throw FormatException('a matrix of ${rows}x$columns cannot be read');
    }

    return DenseMatrix._(rows, columns, reader.float32List(rows * columns));
  }

  @override
  final int rows;
  @override
  final int columns;

  final Float32List _data;

  @override
  void addRowTo(Float32List target, int row) {
    final offset = row * columns;
    for (var i = 0; i < columns; i++) {
      target[i] += _data[offset + i];
    }
  }

  @override
  double dotRow(Float32List vector, int row) {
    final offset = row * columns;
    var sum = 0.0;
    // Two rounding steps, one for the product and one for the sum. Fusing
    // them into a multiply-add would be the faster reading of `d += a * b`,
    // and it is what the C++ is allowed to do, but the reference the goldens
    // come from does not: contraction is optional, and the answers here were
    // measured against a build that keeps both roundings.
    for (var i = 0; i < columns; i++) {
      sum = float32(sum + float32(vector[i] * _data[offset + i]));
    }

    return _requireNumber(sum);
  }
}

/// A matrix compressed with product quantization.
///
/// Each vector is split into [ProductQuantizer.subquantizers] sub-vectors and
/// every sub-vector is replaced by the index of the nearest of 256 centroids.
/// For `lid.176.ftz` that is 8 bytes of codes plus one byte of quantized norm
/// per row: 16 float32 values shrink from 64 bytes to 9.
class QuantizedMatrix implements Matrix {
  QuantizedMatrix._(
    this.rows,
    this.columns,
    this._codes,
    this._quantizer,
    this._normCodes,
    this._normQuantizer,
  );

  factory QuantizedMatrix.read(BinaryReader reader) {
    final hasNorms = reader.boolean();
    final rows = reader.int64();
    final columns = reader.int64();
    final codeSize = reader.int32();
    if (rows < 0 || columns < 0) {
      throw FormatException('a matrix of ${rows}x$columns cannot be read');
    }

    final codes = reader.byteView(codeSize);
    final quantizer = ProductQuantizer.read(reader);
    // fastText sizes the code block as one code per sub-space per row. Any
    // other value means the codes and the codebook describe different
    // matrices, and every lookup below would run off the end of one of them.
    if (codeSize != rows * quantizer.subquantizers) {
      throw FormatException(
        'the quantized matrix carries $codeSize codes, but $rows rows of '
        '${quantizer.subquantizers} sub-spaces need '
        '${rows * quantizer.subquantizers}',
      );
    }

    if (quantizer.dim != columns) {
      throw FormatException(
        'the codebook describes ${quantizer.dim} columns, but the matrix has '
        '$columns',
      );
    }

    Uint8List? normCodes;
    ProductQuantizer? normQuantizer;
    if (hasNorms) {
      normCodes = reader.byteView(rows);
      normQuantizer = ProductQuantizer.read(reader);
    }
    return QuantizedMatrix._(
      rows,
      columns,
      codes,
      quantizer,
      normCodes,
      normQuantizer,
    );
  }

  @override
  final int rows;
  @override
  final int columns;

  final Uint8List _codes;
  final ProductQuantizer _quantizer;
  final Uint8List? _normCodes;
  final ProductQuantizer? _normQuantizer;

  /// The row norm is quantized too, by a separate one-dimensional
  /// quantizer.
  double _norm(int row) {
    final codes = _normCodes;
    if (codes == null) {
      return 1;
    }
    final quantizer = _normQuantizer!;
    return quantizer.centroids[quantizer.centroidOffset(0, codes[row])];
  }

  @override
  void addRowTo(Float32List target, int row) =>
      _quantizer.addCode(target, _codes, row, _norm(row));

  @override
  double dotRow(Float32List vector, int row) =>
      _requireNumber(_quantizer.dotCode(vector, _codes, row, _norm(row)));
}

/// A product quantization codebook: 256 centroids per sub-space.
class ProductQuantizer {
  ProductQuantizer._(
    this.dim,
    this.subquantizers,
    this.subDim,
    this.lastSubDim,
    this.centroids,
  );

  factory ProductQuantizer.read(BinaryReader reader) {
    final dim = reader.int32();
    final subquantizers = reader.int32();
    final subDim = reader.int32();
    final lastSubDim = reader.int32();
    // The sub-spaces tile the vector exactly: every one of them is subDim
    // wide except the last. Holding the file to that is what keeps every
    // centroid offset inside the codebook.
    if (subquantizers < 1 ||
        subDim < 1 ||
        lastSubDim < 1 ||
        (subquantizers - 1) * subDim + lastSubDim != dim) {
      throw FormatException(
        'the codebook does not tile a vector of $dim: $subquantizers '
        'sub-spaces of $subDim, the last one $lastSubDim',
      );
    }

    return ProductQuantizer._(
      dim,
      subquantizers,
      subDim,
      lastSubDim,
      reader.float32List(dim * centroidsPerSubquantizer),
    );
  }

  static const centroidsPerSubquantizer = 256;

  final int dim;
  final int subquantizers;
  final int subDim;

  /// The last sub-space can be shorter when [dim] is not a multiple of
  /// [subDim].
  final int lastSubDim;

  final Float32List centroids;

  int centroidOffset(int subquantizer, int code) =>
      subquantizer == subquantizers - 1
      ? subquantizer * centroidsPerSubquantizer * subDim + code * lastSubDim
      : (subquantizer * centroidsPerSubquantizer + code) * subDim;

  void addCode(Float32List target, Uint8List codes, int row, double scale) {
    final base = subquantizers * row;
    for (var m = 0; m < subquantizers; m++) {
      final centroid = centroidOffset(m, codes[base + m]);
      final width = m == subquantizers - 1 ? lastSubDim : subDim;
      final offset = m * subDim;
      for (var i = 0; i < width; i++) {
        target[offset + i] += scale * centroids[centroid + i];
      }
    }
  }

  double dotCode(Float32List vector, Uint8List codes, int row, double scale) {
    final base = subquantizers * row;
    var sum = 0.0;
    for (var m = 0; m < subquantizers; m++) {
      final centroid = centroidOffset(m, codes[base + m]);
      final width = m == subquantizers - 1 ? lastSubDim : subDim;
      final offset = m * subDim;
      for (var i = 0; i < width; i++) {
        sum = float32(sum + vector[offset + i] * centroids[centroid + i]);
      }
    }

    return float32(sum * scale);
  }
}
