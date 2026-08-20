import 'package:lang_id/lang_id.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// Exercises the whole path — header, dictionary, dense matrix, softmax —
/// without needing downloaded weights or `dart:io`, so it also runs on the
/// web.
void main() {
  group('synthetic model', () {
    late LanguageIdentifier identifier;

    setUpAll(
      () => identifier = LanguageIdentifier.fromBytes(buildSyntheticModel()),
    );

    test('parses the header', () {
      expect(identifier.info.formatVersion, 12);
      expect(identifier.info.quantized, isFalse);
      expect(identifier.info.args.dim, 2);
      expect(identifier.info.args.loss, FastTextLoss.softmax);
      expect(identifier.info.wordCount, 3);
      expect(identifier.info.labelCount, 2);
      expect(identifier.languages, ['x', 'y']);
    });

    test('tells labels apart by word', () {
      expect(identifier.identify('alpha')!.label, 'x');
      expect(identifier.identify('beta')!.label, 'y');
    });

    test('probabilities are normalized', () {
      final result = identifier.predict('alpha', k: 2);
      expect(result, hasLength(2));
      final sum = result.fold<double>(0, (a, p) => a + p.probability);
      // std_log adds 1e-5 under the logarithm, hence the tolerance.
      expect(sum, closeTo(1.0, 1e-4));
    });

    test('unknown words contribute nothing without character n-grams', () {
      // Only the </s> token is left, so both labels are equally likely.
      final result = identifier.predict('unknownword', k: 2);
      expect(result.first.probability, closeTo(0.5, 1e-4));
    });
  });
}
