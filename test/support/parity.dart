import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:lang_id/lang_id.dart';
import 'package:test/test.dart';

/// Whether weights that are not on disk should fail the suite instead of
/// skipping it.
///
/// The models are not in the repository, so on a fresh clone the tests that
/// need them cannot run. Skipping quietly is right for a working copy and
/// wrong everywhere else: it is how the package's main test came to report
/// success without ever running. Continuous integration fetches the weights
/// and sets this, and then a missing file is a failure.
bool get modelsAreRequired =>
    Platform.environment['LANG_ID_REQUIRE_MODELS'] == '1' ||
    Platform.environment['CI'] == 'true';

/// Compares a run against reference answers produced by the original
/// fastText, as exactly as the reference can be reproduced at all.
///
/// Labels and sentence vectors are compared bit for bit with no allowance.
/// Probabilities are too, with one bounded exception: fastText computes its
/// exponentials in `float`, and the platform's `expf` is not always the
/// correctly rounded result — the value Dart produces by rounding a double
/// `exp` is. Where the two disagree it is by one ulp, which the walk down
/// the Huffman tree can grow to a few by the time it reaches a probability.
///
/// The allowance is deliberately narrow, because a mistake in the arithmetic
/// does not look like this. The one this suite caught — reading `d += a * b`
/// as a fused multiply-add, which the reference build does not emit — moved
/// 27 of 41 cases by up to 398 ulps, against the 3 cases and 8 ulps that the
/// libm difference accounts for.
void expectPredictionsMatchReference(
  FastTextClassifier classifier,
  Map<String, dynamic> golden, {
  int allowedInexactCases = 4,
  int allowedUlps = 8,
}) {
  final k = golden['k'] as int;
  final cases = golden['cases'] as List<dynamic>;
  expect(cases, isNotEmpty, reason: 'the golden file has no cases');

  final inexact = <String>[];
  var worstUlps = 0;
  for (final entry in cases) {
    final testCase = entry as Map<String, dynamic>;
    final text = testCase['text'] as String;
    final expected = (testCase['predictions'] as List<dynamic>)
        .cast<Map<String, dynamic>>();
    final actual = classifier.predict(text, k: k);
    final reason = 'text: ${jsonEncode(text)}';

    expect(
      [for (final prediction in actual) prediction.label],
      [for (final prediction in expected) prediction['label']],
      reason: reason,
    );

    for (var i = 0; i < expected.length; i++) {
      final distance = ulpsApart(
        actual[i].probability,
        expected[i]['probability'] as double,
      );
      if (distance == 0) {
        continue;
      }

      worstUlps = distance > worstUlps ? distance : worstUlps;
      inexact.add('$reason, ${actual[i].label}: $distance ulps');
      break;
    }
  }

  expect(
    worstUlps,
    lessThanOrEqualTo(allowedUlps),
    reason:
        'a probability is too far from the reference:\n${inexact.join('\n')}',
  );
  expect(
    inexact,
    hasLength(lessThanOrEqualTo(allowedInexactCases)),
    reason:
        'too many cases fail to match exactly, which is what a mistake in '
        'the arithmetic looks like:\n${inexact.join('\n')}',
  );
}

/// Compares the sentence vector of every case, bit for bit.
void expectSentenceVectorsMatchReference(
  FastTextClassifier classifier,
  Map<String, dynamic> golden,
) {
  for (final entry in golden['cases'] as List<dynamic>) {
    final testCase = entry as Map<String, dynamic>;
    final text = testCase['text'] as String;

    expect(
      hexOf(classifier.sentenceVector(text)),
      testCase['sentenceVector'],
      reason: 'text: ${jsonEncode(text)}',
    );
  }
}

/// How many representable float32 values lie between [a] and [b]; 0 when
/// they are the same value.
int ulpsApart(double a, double b) {
  final values = Float32List(2);
  values[0] = a;
  values[1] = b;
  final bits = Int32List.view(values.buffer);

  return (bits[0] - bits[1]).abs();
}

/// The raw little-endian bytes of a float32 list, so a comparison against
/// them is exact rather than rounded through decimal.
String hexOf(Float32List values) {
  final bytes = Uint8List.view(
    values.buffer,
    values.offsetInBytes,
    values.lengthInBytes,
  );

  return [
    for (final byte in bytes) byte.toRadixString(16).padLeft(2, '0'),
  ].join();
}
