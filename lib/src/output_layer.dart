import 'dart:math' as math;
import 'dart:typed_data';

import 'float32.dart';
import 'matrix.dart';
import 'model_args.dart';

/// The output layer: turns the hidden vector into the top-k labels.
///
/// Its shape is fixed by the loss the model was trained with. The file stores
/// only the weight matrix; everything else is rebuilt from the parameters.
abstract class OutputLayer {
  /// Builds the layer matching the model's loss.
  ///
  /// [labelCounts] matters only to hierarchical softmax, which rebuilds from
  /// it the very same Huffman tree that was used during training.
  static OutputLayer create(
    ModelArgs args,
    Matrix weights,
    List<int> labelCounts,
  ) => switch (args.loss) {
    FastTextLoss.hierarchicalSoftmax => HierarchicalSoftmaxLayer(
      weights,
      labelCounts,
    ),
    FastTextLoss.softmax => SoftmaxLayer(weights, labelCounts.length),
    FastTextLoss.negativeSampling ||
    FastTextLoss.oneVsAll => BinaryLogisticLayer(weights, labelCounts.length),
  };

  int get labelCount;

  /// Label indices and their log scores, highest score first.
  List<ScoredLabel> predict(Float32List hidden, int k, double threshold);
}

/// A label and its log score.
class ScoredLabel {
  const ScoredLabel(this.label, this.score);
  final int label;
  final double score;
}

/// fastText's shifted logarithm: `log(x + 1e-5)`, so that a zero probability
/// does not produce negative infinity. The argument is rounded to float32,
/// which in C++ happens through the conversion to the parameter type.
double _stableLog(double x) => float32(math.log(float32(x) + 1e-5));

/// The sigmoid exactly as the hierarchical softmax tree walk computes it:
/// `1 + exp(-x)` is summed in float32. For large `x` the sum collapses to
/// exactly one and `1 - f` to zero; computed in double it yields one ulp
/// instead of zero, and the log of a rare branch drifts by several digits.
double _sigmoid(double x) =>
    float32(1.0 / float32(1.0 + float32(math.exp(-x))));

/// The table-driven sigmoid: 512 values over [-8, 8], and exact 0 and 1
/// outside that range. This is what negative sampling and one-vs-all use;
/// only hierarchical softmax takes the explicit-formula branch.
double _sigmoidFromTable(double x) {
  if (x < -_maxSigmoid) {
    return 0;
  }

  if (x > _maxSigmoid) {
    return 1;
  }

  final index = float32(
    float32(
          float32(float32(x + _maxSigmoid) * _sigmoidTableSize) / _maxSigmoid,
        ) /
        2,
  ).toInt();

  return _sigmoidTable[index];
}

const _sigmoidTableSize = 512;
const _maxSigmoid = 8;

final Float32List _sigmoidTable = () {
  final table = Float32List(_sigmoidTableSize + 1);
  for (var i = 0; i <= _sigmoidTableSize; i++) {
    final x = float32(
      float32(float32((i * 2 * _maxSigmoid).toDouble()) / _sigmoidTableSize) -
          _maxSigmoid,
    );
    // Here fastText sums in double, unlike the formula above.
    table[i] = 1.0 / (1.0 + math.exp(-x));
  }
  return table;
}();

/// Accumulator for the k best labels: a list kept sorted by descending
/// score. k is a single-digit number here, so insertion sort beats any heap.
class _TopK {
  _TopK(this.k);

  final int k;
  final List<ScoredLabel> _items = [];

  bool get isFull => _items.length >= k;

  double get worstScore => _items.last.score;

  void add(int label, double score) {
    var i = _items.length;
    while (i > 0 && _items[i - 1].score < score) {
      i--;
    }
    _items.insert(i, ScoredLabel(label, score));
    if (_items.length > k) {
      _items.removeLast();
    }
  }

  List<ScoredLabel> get result => _items;
}

/// Hierarchical softmax: labels are the leaves of a Huffman tree and the
/// rows of the weight matrix correspond to its internal nodes. A label's
/// probability is the product of the sigmoids along the path from the root to
/// the leaf, so the walk can be cut off as soon as the accumulated score
/// falls below the k-th best found so far.
class HierarchicalSoftmaxLayer implements OutputLayer {
  HierarchicalSoftmaxLayer(this._weights, List<int> labelCounts)
    : labelCount = labelCounts.length,
      _left = Int32List(2 * labelCounts.length - 1),
      _right = Int32List(2 * labelCounts.length - 1) {
    _buildTree(labelCounts);
  }

  @override
  final int labelCount;

  final Matrix _weights;
  final Int32List _left;
  final Int32List _right;

  /// A literal port of `HierarchicalSoftmaxLoss::buildTree`. The tree has to
  /// match the one used during training down to the last node, or the weights
  /// of the internal nodes stop meaning what they meant.
  void _buildTree(List<int> counts) {
    final nodes = 2 * labelCount - 1;
    final nodeCounts = Float64List(nodes)
      ..fillRange(0, nodes, 1000000000000000);
    for (var i = 0; i < labelCount; i++) {
      nodeCounts[i] = counts[i].toDouble();
    }
    _left.fillRange(0, nodes, -1);
    _right.fillRange(0, nodes, -1);

    var leaf = labelCount - 1;
    var node = labelCount;
    for (var i = labelCount; i < nodes; i++) {
      final children = <int>[];
      for (var c = 0; c < 2; c++) {
        if (leaf >= 0 && nodeCounts[leaf] < nodeCounts[node]) {
          children.add(leaf--);
        } else {
          children.add(node++);
        }
      }
      _left[i] = children[0];
      _right[i] = children[1];
      nodeCounts[i] = nodeCounts[children[0]] + nodeCounts[children[1]];
    }
  }

  @override
  List<ScoredLabel> predict(Float32List hidden, int k, double threshold) {
    final top = _TopK(k);
    _walk(2 * labelCount - 2, 0, hidden, top, _stableLog(threshold));
    return top.result;
  }

  void _walk(
    int node,
    double score,
    Float32List hidden,
    _TopK top,
    double minScore,
  ) {
    if (score < minScore) {
      return;
    }

    if (top.isFull && score < top.worstScore) {
      return;
    }

    if (_left[node] == -1 && _right[node] == -1) {
      top.add(node, score);
      return;
    }

    final p = _sigmoid(_weights.dotRow(hidden, node - labelCount));
    _walk(
      _left[node],
      float32(score + _stableLog(1.0 - p)),
      hidden,
      top,
      minScore,
    );
    _walk(_right[node], float32(score + _stableLog(p)), hidden, top, minScore);
  }
}

/// Plain softmax over all labels.
class SoftmaxLayer implements OutputLayer {
  SoftmaxLayer(this._weights, this.labelCount);

  @override
  final int labelCount;
  final Matrix _weights;

  @override
  List<ScoredLabel> predict(Float32List hidden, int k, double threshold) {
    final scores = Float32List(labelCount);
    for (var i = 0; i < labelCount; i++) {
      scores[i] = _weights.dotRow(hidden, i);
    }
    var max = scores[0];
    for (var i = 0; i < labelCount; i++) {
      if (scores[i] > max) {
        max = scores[i];
      }
    }

    var sum = 0.0;
    for (var i = 0; i < labelCount; i++) {
      scores[i] = math.exp(scores[i] - max);
      sum = float32(sum + scores[i]);
    }

    for (var i = 0; i < labelCount; i++) {
      scores[i] /= sum;
    }

    return _bestOf(scores, k, threshold);
  }
}

/// An independent sigmoid per label: negative sampling and one-vs-all.
class BinaryLogisticLayer implements OutputLayer {
  BinaryLogisticLayer(this._weights, this.labelCount);

  @override
  final int labelCount;
  final Matrix _weights;

  @override
  List<ScoredLabel> predict(Float32List hidden, int k, double threshold) {
    final scores = Float32List(labelCount);
    for (var i = 0; i < labelCount; i++) {
      scores[i] = _sigmoidFromTable(_weights.dotRow(hidden, i));
    }

    return _bestOf(scores, k, threshold);
  }
}

/// Picks the k best out of ready probabilities — the shared tail of the
/// softmax and sigmoid layers.
List<ScoredLabel> _bestOf(Float32List probabilities, int k, double threshold) {
  final top = _TopK(k);
  for (var i = 0; i < probabilities.length; i++) {
    if (probabilities[i] < threshold) {
      continue;
    }

    final score = _stableLog(probabilities[i]);
    if (top.isFull && score < top.worstScore) {
      continue;
    }

    top.add(i, score);
  }
  return top.result;
}
