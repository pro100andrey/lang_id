@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:lang_id/lang_id_io.dart';
import 'package:test/test.dart';

/// Parity against the original fastText on the paths `lid.176` cannot reach.
///
/// `lid.176` is a single hierarchical-softmax model without word n-grams, so
/// on its own it leaves three of the four losses and the whole of
/// `addWordNgrams` uncompared. The models here are trained by the original
/// fastText and committed beside their answers; `tool/generate_fixtures.py`
/// regenerates both.
///
/// The corpus behind them separates its labels by word order alone — `a b a b`
/// against `b a b a` — so the n-gram models cannot be reproduced by a port
/// that ignores wordNgrams.
void main() {
  const models = ['softmax', 'negative_sampling', 'one_vs_all'];

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

      test('predictions match the reference bit for bit', () {
        final k = golden['k'] as int;
        final cases = golden['cases'] as List<dynamic>;
        expect(cases, isNotEmpty);

        for (final entry in cases) {
          final testCase = entry as Map<String, dynamic>;
          final text = testCase['text'] as String;
          final expected = (testCase['predictions'] as List<dynamic>)
              .cast<Map<String, dynamic>>();
          final actual = identifier.predict(text, k: k);
          final reason = 'text: ${jsonEncode(text)}';

          expect(
            [for (final p in actual) p.label],
            [for (final p in expected) p['label']],
            reason: reason,
          );
          for (var i = 0; i < expected.length; i++) {
            expect(
              actual[i].probability,
              expected[i]['probability'],
              reason: '$reason, label ${actual[i].label}',
            );
          }
        }
      });

      test('sentence vectors match the reference bit for bit', () {
        for (final entry in golden['cases'] as List<dynamic>) {
          final testCase = entry as Map<String, dynamic>;
          final text = testCase['text'] as String;

          expect(
            hexOf(identifier.sentenceVector(text)),
            testCase['sentenceVector'],
            reason: 'text: ${jsonEncode(text)}',
          );
        }
      });
    });
  }
}

/// The raw little-endian bytes of a float32 list, so the comparison is exact
/// rather than rounded through decimal.
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
