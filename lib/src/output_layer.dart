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
    // `x` is a float, so C++ picks the float overload of exp here, exactly
    // as it does in the formula above. What differs is the sum: this one is
    // written against a double literal and stays in double.
    table[i] = 1.0 / (1.0 + float32(math.exp(-x)));
  }
  return table;
}();

/// Accumulator for the k best labels, kept the way fastText keeps them.
///
/// C++ holds the candidates in a `std::vector` treated as a heap ordered by
/// `comparePairs`, so the front is the worst of the k found so far, and turns
/// it into an answer with `sort_heap`. Labels that score exactly the same
/// come out in an order that is an artefact of those heap operations, and a
/// list kept sorted by insertion does not reproduce it: on a tie an insert
/// keeps the older entry in front, while the heap does not, and the two
/// disagree about which label wins at k = 1.
///
/// Real models reach this. Two labels of a model trained with negative
/// sampling score bit-identically on an empty line, and the reference returns
/// them the other way round — see test/fixtures.
class _TopK {
  _TopK(this.k);

  final int k;
  final List<ScoredLabel> _items = [];

  bool get isFull => _items.length >= k;

  /// The worst of the scores kept so far: the front of the heap.
  double get worstScore => _items.first.score;

  void add(int label, double score) {
    _items.add(ScoredLabel(label, score));
    _siftUp(_items.length - 1);
    if (_items.length > k) {
      _popFront(_items.length);
      _items.removeLast();
    }
  }

  /// The kept labels, best score first. Consumes the heap, so read it once.
  List<ScoredLabel> get result {
    // `sort_heap`: move the front past the end of the heap, over and over,
    // which leaves the range sorted by the same comparison.
    for (var end = _items.length; end > 1; end--) {
      _popFront(end);
    }

    return _items;
  }

  /// `comparePairs`: with a "greater" comparison the heap is a min-heap by
  /// score, which is what makes the front the first candidate to drop out.
  static bool _scoresAbove(ScoredLabel a, ScoredLabel b) => a.score > b.score;

  /// `std::push_heap`: the last element climbs while its parent scores above
  /// it.
  void _siftUp(int index) {
    final value = _items[index];
    var i = index;
    while (i > 0) {
      final parent = (i - 1) ~/ 2;
      if (!_scoresAbove(_items[parent], value)) {
        break;
      }

      _items[i] = _items[parent];
      i = parent;
    }

    _items[i] = value;
  }

  /// `std::pop_heap` over `[0, end)`: the front leaves the heap and lands
  /// just past its end, which is what makes repeated pops a sort.
  ///
  /// The standard fixes the heap property but not what happens to equal
  /// elements, and the two library implementations take the same route: the
  /// hole from the front walks all the way down to a leaf before the tail
  /// element is put in and sifted back up. Stopping the descent early — the
  /// textbook sift-down — reorders ties differently and disagrees with the
  /// reference on about half of them.
  void _popFront(int end) {
    final front = _items[0];
    final hole = _sinkHoleToLeaf(end);
    final last = end - 1;
    if (hole == last) {
      _items[hole] = front;

      return;
    }

    _items[hole] = _items[last];
    _items[last] = front;
    _siftUp(hole);
  }

  /// Walks the hole left at the front down to a leaf, always following the
  /// child that scores lower, and answers where it stopped.
  int _sinkHoleToLeaf(int end) {
    var hole = 0;
    var child = 0;
    while (true) {
      child = 2 * child + 1;
      if (child + 1 < end && _scoresAbove(_items[child], _items[child + 1])) {
        child++;
      }

      _items[hole] = _items[child];
      hole = child;
      if (child > (end - 2) ~/ 2) {
        return hole;
      }
    }
  }
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
      // Both operands are float32 and C++ keeps the difference there before
      // handing it to exp; the store rounds what comes back.
      scores[i] = math.exp(float32(scores[i] - max));
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
  // C++ takes the threshold as a float, so the comparison below is between
  // two float32 values. Hierarchical softmax rounds it the same way, through
  // the argument of _stableLog; without this the same threshold would mean
  // two slightly different things depending on the loss.
  final cutoff = float32(threshold);
  for (var i = 0; i < probabilities.length; i++) {
    if (probabilities[i] < cutoff) {
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
