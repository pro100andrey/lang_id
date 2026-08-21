@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:lang_id/lang_id_io.dart';
import 'package:test/test.dart';

void main() {
  final modelFile = File('models/lid.176.ftz');
  if (!modelFile.existsSync()) {
    test('no model', () {}, skip: 'run dart run tool/download_model.dart');
    return;
  }

  late FastTextClassifier classifier;
  setUpAll(() => classifier = loadModelSync(modelFile.path));

  group('loading', () {
    test('parses the header of a quantized model', () {
      expect(classifier.info.formatVersion, 12);
      expect(classifier.info.quantized, isTrue);
      expect(classifier.info.labelCount, 176);
      expect(classifier.info.args.dim, 16);
      expect(classifier.info.args.minCharNgram, 2);
      expect(classifier.info.args.maxCharNgram, 4);
      expect(classifier.info.args.loss, FastTextLoss.hierarchicalSoftmax);
      expect(
        classifier.info.args.architecture,
        FastTextArchitecture.supervised,
      );
    });

    test('knows 176 languages', () {
      expect(classifier.labels, hasLength(176));
      expect(classifier.labels, containsAll(['ru', 'en', 'zh', 'ja']));
      expect(
        classifier.labels.first,
        'en',
        reason: 'ordered by descending frequency',
      );
    });

    test('rejects a foreign file', () {
      expect(
        () => FastTextClassifier.fromBytes(Uint8List(64)),
        throwsA(isA<FormatException>()),
      );
    });
  });

  group('classify', () {
    const cases = {
      'Привет, как дела? Сегодня отличная погода.': 'ru',
      'The quick brown fox jumps over the lazy dog': 'en',
      'Bonjour tout le monde, comment allez-vous ?': 'fr',
      'Die Katze sitzt auf der Matte und schläft': 'de',
      '今日はいい天気ですね': 'ja',
      'Мова програмування Dart створена компанією Google': 'uk',
    };

    for (final entry in cases.entries) {
      final title = entry.key.length > 24
          ? '${entry.key.substring(0, 24)}…'
          : entry.key;
      test('«$title» → ${entry.value}', () {
        final result = classifier.classify(entry.key);
        expect(result, isNotNull);
        expect(result!.label, entry.value);
        expect(result.probability, greaterThan(0.5));
      });
    }
  });

  group('sentenceVector', () {
    test('has the model dimension', () {
      expect(
        classifier.sentenceVector('Мова програмування Dart'),
        hasLength(16),
      );
    });

    test('is stable across calls and unaffected by predict', () {
      const text = 'Мова програмування Dart';
      final first = classifier.sentenceVector(text);
      classifier.predict('今日はいい天気ですね', k: 3);
      expect(classifier.sentenceVector(text), first);
    });

    test('returns a fresh list the caller may modify', () {
      const text = 'Мова програмування Dart';
      classifier.sentenceVector(text)[0] = 12345;
      expect(classifier.sentenceVector(text)[0], isNot(12345));
    });

    test('different languages give different vectors', () {
      expect(
        classifier.sentenceVector('The quick brown fox jumps over the dog'),
        isNot(classifier.sentenceVector('Мова програмування Dart')),
      );
    });
  });

  group('predict', () {
    test('returns k candidates, most likely first', () {
      final result = classifier.predict('Привет, как дела?', k: 5);
      expect(result.length, lessThanOrEqualTo(5));
      expect(result.first.label, 'ru');
      for (var i = 1; i < result.length; i++) {
        expect(
          result[i].probability,
          lessThanOrEqualTo(result[i - 1].probability),
        );
      }
    });

    test('the threshold cuts off the unlikely', () {
      final all = classifier.predict('Привет, как дела?', k: 5);
      final filtered = classifier.predict(
        'Привет, как дела?',
        k: 5,
        threshold: 0.5,
      );
      expect(filtered, hasLength(1));
      expect(filtered.first.label, all.first.label);
    });

    test('k below one is an error', () {
      expect(
        () => classifier.predict('text', k: 0),
        throwsA(isA<RangeError>()),
      );
    });

    test('an empty string does not break prediction', () {
      expect(() => classifier.predict(''), returnsNormally);
    });

    test(
      'joinLines folds lines together, otherwise only the first is read',
      () {
        const text = 'yes\nПривет как дела сегодня отличная погода в Москве';
        expect(classifier.classify(text)!.label, 'ru');
        expect(classifier.classify(text, joinLines: false)!.label, isNot('ru'));
      },
    );

    test('buffer reuse does not corrupt neighbouring calls', () {
      final first = classifier.predict('The quick brown fox', k: 3);
      classifier.predict('今日はいい天気ですね', k: 3);
      final again = classifier.predict('The quick brown fox', k: 3);
      expect(again, first);
    });
  });
}
