import 'dart:typed_data';

import 'package:lang_id/lang_id.dart';
import 'package:lang_id/src/binary_reader.dart';
import 'package:lang_id/src/dictionary.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// The dictionary's two hash tables — words by their bytes, and the prune
/// index — checked against the plain reading they replaced. The real models
/// exercise them only through the parity goldens, where a mistake shows up
/// as a wrong number with no hint of where it came from.
void main() {
  /// The header of a model with word n-grams over a table of [bucket]
  /// entries, read back as the dictionary will see it.
  ModelArgs argsWith({required int bucket}) {
    final reader = BinaryReader(
      buildSyntheticModel(wordNgrams: 2, bucket: bucket),
    );
    final version = (reader..int32()).int32();

    return ModelArgs.read(reader, version);
  }

  /// A dictionary section: [words] then [labels], followed by [prune] when
  /// there is one.
  Dictionary dictionaryOf(
    ModelArgs args,
    List<String> words, {
    List<String> labels = const ['__label__x'],
    List<(int, int)>? prune,
  }) {
    final sink = ByteSink()
      ..int32(words.length + labels.length)
      ..int32(words.length)
      ..int32(labels.length)
      ..int64(100)
      ..int64(prune?.length ?? -1);
    for (final word in words) {
      sink
        ..cString(word)
        ..int64(1)
        ..uint8(0);
    }
    for (final label in labels) {
      sink
        ..cString(label)
        ..int64(1)
        ..uint8(1);
    }
    for (final (from, to) in prune ?? const <(int, int)>[]) {
      sink
        ..int32(from)
        ..int32(to);
    }

    return Dictionary.read(BinaryReader(sink.bytes), args);
  }

  /// Enough distinct tokens to give every bucket a fair chance.
  final text = [for (var i = 0; i < 3000; i++) 'w$i'].join(' ');

  group('words by bytes', () {
    test('every word of a large dictionary is found, collisions and all', () {
      // 2,000 words in 4,096 slots: plenty of probes, and some of them wrap
      // past the last slot to the first.
      final args = argsWith(bucket: 10);
      final words = ['</s>', for (var i = 0; i < 2000; i++) 'word$i'];
      final dictionary = dictionaryOf(args, words);

      for (var i = 1; i < words.length; i++) {
        // A known word comes back as its own row first, then its word
        // n-gram with </s> — which, with word n-grams on, is hashed.
        expect(dictionary.lineToIndices(words[i]).first, i);
      }
      expect(dictionary.lineToIndices('unknown').first, 0);
    });

    test('a word listed twice answers with the later index, as a map did', () {
      final dictionary = dictionaryOf(argsWith(bucket: 10), [
        '</s>',
        'twice',
        'other',
        'twice',
      ]);

      expect(dictionary.lineToIndices('twice').first, 3);
      expect(dictionary.lineToIndices('other').first, 2);
    });
  });

  group('prune index', () {
    const bucket = 4096;

    test('maps buckets the way a map would, collisions and all', () {
      // Every 64th bucket is kept: 64 entries in 128 slots, with keys up to
      // 4,096 — so nearly every key lands on a slot taken by another, and
      // some probes wrap. A kept bucket b becomes row b ~/ 64.
      final args = argsWith(bucket: bucket);
      final words = ['</s>'];
      final full = dictionaryOf(args, words);
      final pruned = dictionaryOf(
        args,
        words,
        prune: [
          for (var b = 5; b < bucket; b += 64) (b, b ~/ 64),
        ],
      );

      final expected = [
        for (final index in full.lineToIndices(text))
          if (index < words.length)
            index
          else if ((index - words.length) % 64 == 5)
            words.length + (index - words.length) ~/ 64,
      ];
      expect(expected.length, greaterThan(words.length + 10));
      expect(pruned.lineToIndices(text), expected);
      expect(pruned.inputRowsNeeded, words.length + (bucket - 1) ~/ 64 + 1);
    });

    test('a key written twice keeps its last value, in rows needed too', () {
      final args = argsWith(bucket: bucket);
      final dictionary = dictionaryOf(
        args,
        ['</s>'],
        prune: [(7, 5000), (7, 3)],
      );

      expect(dictionary.inputRowsNeeded, 1 + 3 + 1);
    });

    test('a negative row is refused', () {
      expect(
        () =>
            dictionaryOf(argsWith(bucket: bucket), ['</s>'], prune: [(7, -1)]),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('prune index maps'),
          ),
        ),
      );
    });

    test('a negative bucket is never looked up and does no harm', () {
      final args = argsWith(bucket: bucket);
      final without = dictionaryOf(args, ['</s>'], prune: [(5, 0)]);
      final with_ = dictionaryOf(args, ['</s>'], prune: [(5, 0), (-3, 1)]);

      expect(with_.lineToIndices(text), without.lineToIndices(text));
    });
  });

  group('prune index size', () {
    // The size is an int64. Its product with 8 wraps on the VM, and the
    // values below used to slip through the bounds check into a table that
    // hung (2^61 + 1), ran out of memory (2^60) or spun on its seventeenth
    // key (2^62). All of them are far larger than the file.
    final model = buildSyntheticModel();
    for (final (name, high, low) in [
      ('2^60', 0x10000000, 0),
      ('2^61 + 1', 0x20000000, 1),
      ('2^62', 0x40000000, 0),
    ]) {
      test('$name is refused before anything is allocated', () {
        final corrupt = Uint8List.fromList(model);
        ByteData.view(corrupt.buffer)
          ..setUint32(syntheticHeaderOffsets.pruneIndexSize, low, Endian.little)
          ..setUint32(
            syntheticHeaderOffsets.pruneIndexSize + 4,
            high,
            Endian.little,
          );

        expect(
          () => FastTextClassifier.fromBytes(corrupt),
          throwsA(
            isA<FormatException>().having(
              (e) => e.message,
              'message',
              contains('prune index size'),
            ),
          ),
        );
      });
    }
  });
}
