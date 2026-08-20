import 'dart:typed_data';

/// A cursor over the fastText binary format.
///
/// The format is a raw dump of C++ structs in host byte order, which means
/// little-endian on every platform fastText is actually built for. On a
/// big-endian host float arrays are read element by element.
///
/// 64-bit fields are assembled from two 32-bit words: on the web `int` is a
/// double and [ByteData.getInt64] throws `UnsupportedError` there. Every real
/// value (token counts, matrix dimensions) fits in 2^53, so the join is exact.
class BinaryReader {
  BinaryReader(this.bytes)
    : _data = .view(bytes.buffer, bytes.offsetInBytes, bytes.length);

  final Uint8List bytes;
  final ByteData _data;
  var _offset = 0;

  static final _hostIsLittleEndian = Endian.host == Endian.little;

  int get offset => _offset;
  int get remaining => bytes.length - _offset;

  int int32() {
    final value = _data.getInt32(_offset, .little);
    _offset += 4;

    return value;
  }

  int int64() {
    final low = _data.getUint32(_offset, .little);
    final high = _data.getInt32(_offset + 4, .little);
    _offset += 8;

    return high * 0x100000000 + low;
  }

  int uint8() => bytes[_offset++];

  bool boolean() => bytes[_offset++] != 0;

  double float64() {
    final value = _data.getFloat64(_offset, .little);
    _offset += 8;

    return value;
  }

  Float32List float32List(int count) {
    final result = Float32List(count);
    if (_hostIsLittleEndian) {
      Uint8List.view(result.buffer).setRange(0, count * 4, bytes, _offset);
      _offset += count * 4;
    } else {
      for (var i = 0; i < count; i++) {
        result[i] = _data.getFloat32(_offset, .little);
        _offset += 4;
      }
    }

    return result;
  }

  /// A window over the source buffer — no copying.
  Uint8List byteView(int count) {
    final view = Uint8List.sublistView(bytes, _offset, _offset + count);
    _offset += count;

    return view;
  }

  /// A null-terminated string, returned as raw bytes. fastText compares words
  /// byte by byte, so they must not be decoded as UTF-8 for lookup.
  Uint8List cString() {
    final start = _offset;
    while (bytes[_offset] != 0) {
      _offset++;
    }

    final view = Uint8List.sublistView(bytes, start, _offset);
    _offset++; // null terminator

    return view;
  }
}
