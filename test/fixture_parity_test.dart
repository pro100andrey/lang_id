@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';

import 'package:lang_id/lang_id_io.dart';
import 'package:test/test.dart';

import 'support/parity.dart';

/// Parity against the original fastText on the paths `lid.176` cannot reach.
///
/// `lid.176` is one hierarchical-softmax model without word n-grams, so on
/// its own it leaves three of the four losses, and the whole of
/// `addWordNgrams`, with nothing to compare against. The models here are
/// trained by the original fastText and committed beside its answers;
/// tool/generate_fixtures.py regenerates both. They are a few kilobytes each,
/// so unlike the weights they live in the repository and always run.
///
/// The corpus behind them separates its labels by word order alone — `a b a
/// b` against `b a b a` — which no model can learn from single words. A port
/// that ignores wordNgrams cannot reproduce these answers.
void main() {
  const models = [
    'ngrams_hs',
    'ngrams_softmax',
    'softmax',
    'negative_sampling',
    'one_vs_all',
  ];

  for (final name in models) {
    group(name, () {
      late LanguageIdentifier identifier;
      late Map<String, dynamic> golden;

      setUpAll(() {
        identifier = loadModelSync('test/fixtures/$name.bin');
        golden =
            jsonDecode(File('test/golden/$name.json').readAsStringSync())
                as Map<String, dynamic>;
      });

      test('predictions match the reference', () {
        expectPredictionsMatchReference(
          identifier,
          golden,
          allowedInexactCases: 1,
        );
      });

      test('sentence vectors match the reference bit for bit', () {
        expectSentenceVectorsMatchReference(identifier, golden);
      });
    });
  }
}
