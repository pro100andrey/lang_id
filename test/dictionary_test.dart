import 'dart:convert';
import 'dart:typed_data';

import 'package:lang_id/src/dictionary.dart';
import 'package:test/test.dart';

void main() {
  group('Dictionary.hash', () {
    // Computed by an independent FNV-1a implementation with the byte cast to
    // a signed int8, exactly the way Dictionary::hash does it in fastText.
    const expected = {
      '': 2166136261,
      'the': 3020861980,
      '</s>': 3617362777,
      '<th': 1619151151,
      'привет': 4033455999,
      '日本': 4164387409,
      '<a>': 2359087600,
      'ÿ': 788157111,
    };

    for (final entry in expected.entries) {
      test('«${entry.key}»', () {
        final bytes = Uint8List.fromList(utf8.encode(entry.key));
        expect(Dictionary.hash(bytes, 0, bytes.length), entry.value);
      });
    }

    test('multiplication mod 2^32 keeps the high bits', () {
      // Bytes with the high bit set are sign-extended, so the result must
      // stay inside the uint32 range and differ from the unsigned reading.
      final signed = Uint8List.fromList([0xFF]);
      final unsignedLike = Uint8List.fromList([0x7F]);
      final a = Dictionary.hash(signed, 0, 1);
      final b = Dictionary.hash(unsignedLike, 0, 1);
      expect(a, isNot(b));
      expect(a, inInclusiveRange(0, 0xFFFFFFFF));
      expect(b, inInclusiveRange(0, 0xFFFFFFFF));
    });

    test('hashes only the given range', () {
      final bytes = Uint8List.fromList(utf8.encode('xxtheyy'));
      final slice = Uint8List.fromList(utf8.encode('the'));
      expect(
        Dictionary.hash(bytes, 2, 5),
        Dictionary.hash(slice, 0, slice.length),
      );
    });
  });
}
