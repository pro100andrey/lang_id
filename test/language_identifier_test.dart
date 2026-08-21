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

  late LanguageIdentifier identifier;
  setUpAll(() => identifier = loadModelSync(modelFile.path));

  group('loading', () {
    test('parses the header of a quantized model', () {
      expect(identifier.info.formatVersion, 12);
      expect(identifier.info.quantized, isTrue);
      expect(identifier.info.labelCount, 176);
      expect(identifier.info.args.dim, 16);
      expect(identifier.info.args.minCharNgram, 2);
      expect(identifier.info.args.maxCharNgram, 4);
      expect(identifier.info.args.loss, FastTextLoss.hierarchicalSoftmax);
      expect(
        identifier.info.args.architecture,
        FastTextArchitecture.supervised,
      );
    });

    test('knows 176 languages', () {
      expect(identifier.labels, hasLength(176));
      expect(identifier.labels, containsAll(['ru', 'en', 'zh', 'ja']));
      expect(
        identifier.labels.first,
        'en',
        reason: 'ordered by descending frequency',
      );
    });

    test('rejects a foreign file', () {
      expect(
        () => LanguageIdentifier.fromBytes(Uint8List(64)),
        throwsA(isA<FormatException>()),
      );
    });
  });

  group('identify', () {
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
        final result = identifier.identify(entry.key);
        expect(result, isNotNull);
        expect(result!.label, entry.value);
        expect(result.probability, greaterThan(0.5));
      });
    }
  });

  group('sentenceVector', () {
    test('has the model dimension', () {
      expect(
        identifier.sentenceVector('Мова програмування Dart'),
        hasLength(16),
      );
    });

    test('is stable across calls and unaffected by predict', () {
      const text = 'Мова програмування Dart';
      final first = identifier.sentenceVector(text);
      identifier.predict('今日はいい天気ですね', k: 3);
      expect(identifier.sentenceVector(text), first);
    });

    test('returns a fresh list the caller may modify', () {
      const text = 'Мова програмування Dart';
      identifier.sentenceVector(text)[0] = 12345;
      expect(identifier.sentenceVector(text)[0], isNot(12345));
    });

    test('different languages give different vectors', () {
      expect(
        identifier.sentenceVector('The quick brown fox jumps over the dog'),
        isNot(identifier.sentenceVector('Мова програмування Dart')),
      );
    });
  });

  group('predict', () {
    test('returns k candidates, most likely first', () {
      final result = identifier.predict('Привет, как дела?', k: 5);
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
      final all = identifier.predict('Привет, как дела?', k: 5);
      final filtered = identifier.predict(
        'Привет, как дела?',
        k: 5,
        threshold: 0.5,
      );
      expect(filtered, hasLength(1));
      expect(filtered.first.label, all.first.label);
    });

    test('k below one is an error', () {
      expect(
        () => identifier.predict('text', k: 0),
        throwsA(isA<RangeError>()),
      );
    });

    test('an empty string does not break prediction', () {
      expect(() => identifier.predict(''), returnsNormally);
    });

    test(
      'joinLines folds lines together, otherwise only the first is read',
      () {
        const text = 'yes\nПривет как дела сегодня отличная погода в Москве';
        expect(identifier.identify(text)!.label, 'ru');
        expect(identifier.identify(text, joinLines: false)!.label, isNot('ru'));
      },
    );

    test('buffer reuse does not corrupt neighbouring calls', () {
      final first = identifier.predict('The quick brown fox', k: 3);
      identifier.predict('今日はいい天気ですね', k: 3);
      final again = identifier.predict('The quick brown fox', k: 3);
      expect(again, first);
    });
  });
}
