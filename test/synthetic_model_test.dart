import 'package:lang_id/lang_id.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// Exercises the whole path — header, dictionary, dense matrix, softmax —
/// without needing downloaded weights or `dart:io`, so it also runs on the
/// web.
void main() {
  group('Prediction', () {
    const likely = Prediction('en', 0.9);
    const unlikely = Prediction('ru', 0.1);

    test('sorts ascending, the way Comparable asks', () {
      // It used to sort descending, which reads well next to predict and
      // means the wrong thing to everything else that takes a Comparable.
      expect(([likely, unlikely]..sort()).last, likely);
      expect(likely.compareTo(unlikely), greaterThan(0));
    });

    test('equally likely labels are ordered by name, never as equal', () {
      const first = Prediction('be', 0.5);
      const second = Prediction('uk', 0.5);

      expect(first.compareTo(second), lessThan(0));
      expect(first.compareTo(first), 0);
    });
  });

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

    group('blank text', () {
      for (final text in ['', ' ', '   \t\n', '\r\n']) {
        test('identify(${text.length} blanks) is null', () {
          expect(identifier.identify(text), isNull);
        });
      }

      test('predict still answers, the way fastText does', () {
        // fastText appends a newline before parsing, which yields the
        // end-of-sentence token, which is always in the dictionary. So there
        // is something to score even for an empty line, and predict says so;
        // only identify calls that nothing.
        expect(identifier.predict(''), isNotEmpty);
      });

      test('text that only looks blank is not', () {
        expect(identifier.identify('\u00a0'), isNotNull);
      });
    });

    test('k of -1 asks for every label, as it does in fastText', () {
      expect(identifier.predict('alpha', k: -1), hasLength(2));
      expect(identifier.predict('alpha'), hasLength(1));
    });

    test('any other k below one is still an error', () {
      expect(() => identifier.predict('alpha', k: 0), throwsRangeError);
      expect(() => identifier.predict('alpha', k: -2), throwsRangeError);
    });

    test('unknown words contribute nothing without character n-grams', () {
      // Only the </s> token is left, so both labels are equally likely.
      final result = identifier.predict('unknownword', k: 2);
      expect(result.first.probability, closeTo(0.5, 1e-4));
    });
  });
}
