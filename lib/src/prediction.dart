import 'language_identifier.dart';

/// A label and its probability.
class Prediction implements Comparable<Prediction> {
  /// Pairs a [label] with the [probability] the model gave it.
  const Prediction(this.label, this.probability);

  /// Language code (`ru`, `en`, `zh`) with the `__label__` prefix stripped.
  final String label;

  /// Probability between 0 and 1.
  ///
  /// It can come out a hair above 1 — around `1.0001` on a certain answer.
  /// fastText adds `1e-5` under every logarithm so that a zero probability
  /// stays finite, and that shift survives into the result. The value is
  /// reproduced the way the original computes it rather than clamped, so that
  /// answers match fastText exactly; clamp it yourself if you would rather
  /// show a percentage that never exceeds 100.
  final double probability;

  /// Orders by [probability], least likely first, and by [label] when two
  /// are equally likely.
  ///
  /// This is the ascending order [Comparable] asks for, so `sort()` leaves
  /// the likeliest last. It used to be the other way round, which read well
  /// at a call site and lied to everything that takes a [Comparable] at its
  /// word. You rarely need it: [LanguageIdentifier.predict] already answers
  /// most likely first.
  @override
  int compareTo(Prediction other) {
    final byProbability = probability.compareTo(other.probability);

    return byProbability != 0 ? byProbability : label.compareTo(other.label);
  }

  @override
  String toString() => '$label ${(probability * 100).toStringAsFixed(1)}%';

  @override
  bool operator ==(Object other) =>
      other is Prediction &&
      other.label == label &&
      other.probability == probability;

  @override
  int get hashCode => Object.hash(label, probability);
}
