import 'dart:typed_data';

import 'package:lang_id/src/binary_reader.dart';
import 'package:test/test.dart';

/// Deliberately not marked `@TestOn('vm')`: 64-bit fields are assembled from
/// two 32-bit words precisely so the package works on the web, where
/// `ByteData.getInt64` is unavailable.
void main() {
  Uint8List bytesOf(List<int> values) => Uint8List.fromList(values);

  group('BinaryReader', () {
    test('reads signed 32-bit numbers', () {
      final reader = BinaryReader(
        bytesOf([0x0c, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]),
      );
      expect(reader.int32(), 12);
      expect(reader.int32(), -1);
    });

    test('assembles 64-bit numbers from two words', () {
      final reader = BinaryReader(
        bytesOf([
          0x7e, 0x85, 0x96, 0x21, 0, 0, 0, 0, // 563512702
          0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // -1
          0, 0, 0, 0, 1, 0, 0, 0, // 2^32
        ]),
      );
      expect(reader.int64(), 563512702);
      expect(reader.int64(), -1);
      expect(reader.int64(), 4294967296);
    });

    test('reads a string up to the null byte', () {
      final reader = BinaryReader(bytesOf([0x74, 0x68, 0x65, 0, 0x61, 0]));
      expect(reader.cString(), bytesOf([0x74, 0x68, 0x65]));
      expect(reader.cString(), bytesOf([0x61]));
      expect(reader.remaining, 0);
    });

    test('reads float32 in bulk with the right byte order', () {
      final source = Float32List.fromList([1.5, -2.25, 3.75]);
      final reader = BinaryReader(Uint8List.view(source.buffer));
      expect(reader.float32List(3), [1.5, -2.25, 3.75]);
    });

    test('reading past the end is a FormatException, not a RangeError', () {
      expect(() => BinaryReader(Uint8List(3)).int32(), throwsFormatException);
      expect(() => BinaryReader(Uint8List(4)).int64(), throwsFormatException);
      expect(() => BinaryReader(Uint8List(0)).uint8(), throwsFormatException);
      expect(() => BinaryReader(Uint8List(0)).boolean(), throwsFormatException);
      expect(() => BinaryReader(Uint8List(4)).float64(), throwsFormatException);
      expect(
        () => BinaryReader(Uint8List(8)).float32List(3),
        throwsFormatException,
      );
      expect(
        () => BinaryReader(Uint8List(2)).byteView(3),
        throwsFormatException,
      );
    });

    test('a count from a corrupt header cannot allocate', () {
      // The size fields are just numbers in the file: a big one has to fail
      // against what is left rather than try to allocate gigabytes.
      expect(
        () => BinaryReader(Uint8List(16)).float32List(0x7FFFFFFF),
        throwsFormatException,
      );
      expect(
        () => BinaryReader(Uint8List(16)).float32List(-1),
        throwsFormatException,
      );
    });

    test('an unterminated string is a FormatException', () {
      expect(
        () => BinaryReader(bytesOf([0x74, 0x68, 0x65])).cString(),
        throwsFormatException,
      );
    });

    test('the offset advances by exactly what was read', () {
      final reader = BinaryReader(Uint8List(32))
        ..int32()
        ..int64()
        ..uint8();
      expect(reader.offset, 13);
      expect(reader.remaining, 19);
    });
  });
}
