import 'dart:typed_data';

import 'package:lang_id/lang_id.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// A model file is untrusted input. These are the ways a corrupt one used to
/// get through: an allocation sized from a raw header field, a shape nobody
/// compared against the header, or a mismatch that produced a confident
/// answer computed from half a vector.
void main() {
  final model = buildSyntheticModel();

  Matcher throwsFormat(String fragment) => throwsA(
    isA<FormatException>().having(
      (e) => e.message,
      'message',
      contains(fragment),
    ),
  );

  void expectRejected(Uint8List corrupt, String fragment) => expect(
    () => FastTextClassifier.fromBytes(corrupt),
    throwsFormat(fragment),
  );

  group('dictionary counts', () {
    test('a negative size is rejected', () {
      expectRejected(
        patchInt32(model, syntheticHeaderOffsets.size, -1),
        'negative size',
      );
    });

    test('counts that do not add up are rejected', () {
      expectRejected(
        patchInt32(model, syntheticHeaderOffsets.wordCount, 2),
        'do not add up',
      );
    });

    test('a size larger than the file is rejected before allocating', () {
      // Four edited bytes used to ask for a multi-gigabyte allocation, which
      // took eleven seconds to fail. The check is against what is left in
      // the buffer, so this returns immediately.
      const huge = 0x7FFFFFFF;
      expectRejected(
        patchInt32(
          patchInt32(model, syntheticHeaderOffsets.size, huge),
          syntheticHeaderOffsets.wordCount,
          huge - 2,
        ),
        'do not fit',
      );
    });

    test('a prune index larger than the file is rejected', () {
      expectRejected(
        patchInt32(model, syntheticHeaderOffsets.pruneIndexSize, 0x7FFFFFF0),
        'prune index size',
      );
    });

    test('a model with no labels is rejected', () {
      expectRejected(buildSyntheticModel(labelCount: 0), 'no labels');
    });
  });

  group('n-gram table', () {
    test('word n-grams with nowhere to put them are refused', () {
      // fastText zeroes the table itself when it trains a model that needs
      // none, so an empty one is ordinary — and an empty one beside n-grams
      // is a file that contradicts itself. Dropping them quietly would leave
      // the model answering from whatever was left.
      expectRejected(
        buildSyntheticModel(wordNgrams: 2),
        'n-gram table of 0 entries',
      );
    });

    test('character n-grams with nowhere to put them are refused', () {
      expectRejected(
        patchInt32(model, syntheticHeaderOffsets.maxCharNgram, 4),
        'n-gram table of 0 entries',
      );
    });

    test('a negative table is refused', () {
      expectRejected(
        patchInt32(model, syntheticHeaderOffsets.bucket, -1),
        'n-gram table of -1 entries',
      );
    });

    test('an empty table on its own is ordinary', () {
      expect(
        FastTextClassifier.fromBytes(model).info.args.bucket,
        0,
        reason: 'what fastText writes for a model with no n-grams at all',
      );
    });
  });

  group('shapes', () {
    test('a dimension the matrices do not have is rejected', () {
      // The quiet case: the header claims four columns, the input matrix has
      // two, and the tail of the hidden vector stays zero while predict
      // answers as if nothing happened.
      expectRejected(
        patchInt32(model, syntheticHeaderOffsets.dim, 4),
        'vectors are 4 wide',
      );
    });

    test('a truncated model is rejected', () {
      expectRejected(
        Uint8List.sublistView(model, 0, model.length - 4),
        'truncated',
      );
    });

    test('a foreign file is rejected', () {
      expect(
        () => FastTextClassifier.fromBytes(Uint8List(64)),
        throwsFormat('wrong file signature'),
      );
    });
  });

  group('weights that are not numbers', () {
    // A NaN passes every comparison as false, so nothing downstream notices
    // it: the sigmoid's range checks let it reach a conversion that throws
    // an unrelated error, and the pruning that keeps predictions ordered
    // stops firing. fastText raises on it and so does this.
    for (final poison in [double.nan, double.infinity]) {
      test('$poison in a weight is refused', () {
        final classifier = FastTextClassifier.fromBytes(
          buildSyntheticModel(poisonedWeight: poison),
        );

        expect(
          () => classifier.predict('alpha'),
          throwsFormat('weights contain NaN or an infinity'),
        );
      });
    }

    test('the order is never quietly wrong instead', () {
      // The failure this replaces: hierarchical softmax used to answer with
      // the likeliest label last.
      final classifier = FastTextClassifier.fromBytes(
        buildSyntheticModel(
          loss: lossHierarchicalSoftmax,
          poisonedWeight: 0 / 0,
        ),
      );

      expect(() => classifier.predict('alpha', k: 4), throwsFormatException);
    });
  });

  test('the model still loads and predicts when nothing is corrupt', () {
    final classifier = FastTextClassifier.fromBytes(model);
    expect(classifier.classify('alpha')!.label, 'x');
  });
}
