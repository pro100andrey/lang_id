import 'dart:convert';
import 'dart:typed_data';

import 'binary_reader.dart';
import 'model_args.dart';

/// The model dictionary: words, labels and the rules that turn a line of
/// text into vector indices.
///
/// This is where all of fastText's "text" side lives — the hash, character
/// n-grams, tokenization. Everything works on UTF-8 bytes, because fastText
/// compares and hashes words byte by byte and cuts n-grams on UTF-8
/// character boundaries.
class Dictionary {
  Dictionary._(
    this._args,
    this.size,
    this.wordCount,
    this.labelCount,
    this.tokenCount,
    this._words,
    this._counts,
    this._types,
    this._index,
    this._pruneIndexSize,
    this._pruneIndex,
  ) : _subwordCache = List<List<int>?>.filled(wordCount, null);

  factory Dictionary.read(BinaryReader reader, ModelArgs args) {
    final size = reader.int32();
    final wordCount = reader.int32();
    final labelCount = reader.int32();
    final tokenCount = reader.int64();
    final pruneIndexSize = reader.int64();
    _validate(reader, size, wordCount, labelCount, pruneIndexSize);

    final words = <Uint8List>[];
    final counts = List<int>.filled(size, 0);
    final types = Uint8List(size);
    final index = <String, int>{};
    for (var i = 0; i < size; i++) {
      final word = reader.cString();
      words.add(word);
      counts[i] = reader.int64();
      types[i] = reader.uint8(); // entry_type is exactly one byte
      index[String.fromCharCodes(word)] = i;
    }

    final pruneIndex = <int, int>{};
    for (var i = 0; i < pruneIndexSize; i++) {
      final from = reader.int32();
      final to = reader.int32();
      pruneIndex[from] = to;
    }

    return Dictionary._(
      args,
      size,
      wordCount,
      labelCount,
      tokenCount,
      words,
      counts,
      types,
      index,
      pruneIndexSize,
      pruneIndex,
    );
  }

  /// Checks the counts before anything is allocated from them.
  ///
  /// They are int32 fields of an untrusted file, and they size every list
  /// here: unchecked, four edited bytes ask for a multi-gigabyte allocation
  /// long before the read runs off the end of the buffer.
  ///
  /// The equality is fastText's own invariant — `size_` is incremented
  /// together with either `nwords_` or `nlabels_` — and holding the reader to
  /// it is what keeps [labelCount] and [labelCounts] from ever describing
  /// different things.
  static void _validate(
    BinaryReader reader,
    int size,
    int wordCount,
    int labelCount,
    int pruneIndexSize,
  ) {
    if (size < 0 || wordCount < 0 || labelCount < 0) {
      throw FormatException(
        'the dictionary has a negative size: $size entries, '
        '$wordCount words, $labelCount labels',
      );
    }

    if (wordCount + labelCount != size) {
      throw FormatException(
        'the dictionary is inconsistent: $wordCount words plus $labelCount '
        'labels do not add up to $size entries',
      );
    }

    // The shortest possible entry is an empty word: one null byte, an int64
    // count and the one-byte type.
    if (size * _minimumEntryBytes > reader.remaining) {
      throw FormatException(
        'the dictionary claims $size entries, which do not fit in the '
        '${reader.remaining} bytes left',
      );
    }

    if (pruneIndexSize < -1 || pruneIndexSize * 8 > reader.remaining) {
      throw FormatException(
        'the prune index size is out of range: '
        '$pruneIndexSize',
      );
    }
  }

  static const _minimumEntryBytes = 10;

  /// The default label prefix. It is not stored in the model file; fastText
  /// uses this same value at training time unless told otherwise.
  static const labelPrefix = '__label__';

  /// End-of-sentence token. The fastText Python binding appends `\n` to the
  /// text before predicting, so this token is always part of the parse.
  static const endOfSentence = '</s>';

  static const _typeLabel = 1;
  static const _bow = 0x3c; // '<'
  static const _eow = 0x3e; // '>'

  final ModelArgs _args;

  /// Total entries: [wordCount] words plus [labelCount] labels.
  final int size;
  final int wordCount;
  final int labelCount;

  /// Number of tokens in the training corpus.
  final int tokenCount;

  final List<Uint8List> _words;
  final List<int> _counts;
  final Uint8List _types;

  /// Word to index. The key is the raw bytes with one byte per code unit,
  /// which is a bijection — unlike UTF-8 decoding, where malformed bytes
  /// would collapse into U+FFFD and produce false matches.
  final Map<String, int> _index;

  /// The prune map size as stored in the file: -1 means the model is not
  /// pruned, 0 means it is pruned down to nothing (every character n-gram is
  /// dropped), and a positive value is the size of the remapping.
  final int _pruneIndexSize;

  /// Remapping of n-gram buckets after pruning (quantized models only).
  final Map<int, int> _pruneIndex;

  final List<List<int>?> _subwordCache;
  var _wrapBuffer = Uint8List(64);

  /// `true` when the dictionary has been pruned, as in `lid.176.ftz`.
  bool get isPruned => _pruneIndexSize >= 0;

  /// Model labels in index order, with the `__label__` prefix.
  List<String> get labels => [
    for (var i = wordCount; i < size; i++)
      utf8.decode(_words[i], allowMalformed: true),
  ];

  /// Labels without the prefix: for `lid.176` these are language codes
  /// such as `ru`, `en`, `zh`.
  List<String> get labelNames => [
    for (final label in labels)
      if (label.startsWith(labelPrefix))
        label.substring(labelPrefix.length)
      else
        label,
  ];

  /// Label frequencies in the training corpus, from which the hierarchical
  /// softmax Huffman tree is rebuilt.
  List<int> get labelCounts => _counts.sublist(wordCount, size);

  /// FNV-1a exactly as fastText computes it: each byte is cast to a signed
  /// `int8_t`, so high UTF-8 bytes are sign-extended.
  static int hash(Uint8List data, int start, int end) {
    var h = 2166136261;
    for (var i = start; i < end; i++) {
      final byte = data[i];
      h = (h ^ (byte <= 127 ? byte : 0xFFFFFF00 | byte)) & 0xFFFFFFFF;
      h = _multiplyMod32(h, 16777619);
    }

    return h;
  }

  /// `a * b mod 2^32`. On the web `int` is a double, and the product of two
  /// 32-bit numbers does not fit in a 53-bit mantissa, so multiply by
  /// halves.
  static int _multiplyMod32(int a, int b) {
    final low = (a & 0xFFFF) * b;
    final high = ((a >>> 16) * b) & 0xFFFF;
    return (low + (high << 16)) & 0xFFFFFFFF;
  }

  /// Turns a line of text into input-matrix row indices — the equivalent of
  /// `Dictionary::getLine` for supervised models.
  ///
  /// With [joinLines], newlines count as ordinary separators; otherwise the
  /// parse stops at the first one, the way the fastText CLI behaves.
  List<int> lineToIndices(String text, {bool joinLines = true}) {
    final bytes = _encodeLine(text, joinLines);
    final indices = <int>[];
    // Only a model trained with wordNgrams > 1 has any, and hashing every
    // token is not worth doing for the models that do not.
    final wordHashes = _args.wordNgrams > 1 ? <int>[] : null;
    var i = 0;
    while (i < bytes.length) {
      final byte = bytes[i];
      if (byte == 0x0a) {
        _addToken(indices, _eosBytes, 0, _eosBytes.length, wordHashes);
        break;
      }

      if (_isSeparator(byte)) {
        i++;
        continue;
      }

      final start = i;
      while (i < bytes.length && !_isSeparator(bytes[i])) {
        i++;
      }

      _addToken(indices, bytes, start, i, wordHashes);
      if (_equals(bytes, start, i, _eosBytes)) {
        break;
      }
    }

    if (wordHashes != null) {
      _addWordNgrams(indices, wordHashes);
    }

    return indices;
  }

  /// `Dictionary::addWordNgrams`: every run of up to [ModelArgs.wordNgrams]
  /// neighbouring words is folded into one hash and lands in the same table
  /// the character n-grams use.
  ///
  /// This is what lets a model tell `a b` from `b a`, which single words
  /// cannot. fastText's own text-classification tutorial recommends training
  /// with `-wordNgrams 2`, so a model that needs this is the ordinary case
  /// rather than an exotic one.
  void _addWordNgrams(List<int> out, List<int> hashes) {
    // A model trained without a hash table has nowhere to put them, exactly
    // as with the character n-grams above.
    if (_args.bucket <= 0) {
      return;
    }

    final reach = _args.wordNgrams;
    final chained = _Uint64();
    for (var i = 0; i < hashes.length; i++) {
      chained.setSignExtended(hashes[i]);
      for (var j = i + 1; j < hashes.length && j < i + reach; j++) {
        chained.multiplyAdd(_wordNgramMultiplier, hashes[j]);
        _pushHash(out, chained.modulo(_args.bucket));
      }
    }
  }

  /// The constant fastText chains word hashes with.
  static const _wordNgramMultiplier = 116049371;

  static final _eosBytes = Uint8List.fromList(utf8.encode(endOfSentence));

  static bool _isSeparator(int byte) =>
      byte == 0x20 || // space
      byte == 0x0a || // \n
      byte == 0x0d || // \r
      byte == 0x09 || // \t
      byte == 0x0b || // \v
      byte == 0x0c || // \f
      byte == 0x00;

  static bool _equals(Uint8List data, int start, int end, Uint8List other) {
    if (end - start != other.length) {
      return false;
    }

    for (var i = 0; i < other.length; i++) {
      if (data[start + i] != other[i]) {
        return false;
      }
    }

    return true;
  }

  Uint8List _encodeLine(String text, bool joinLines) {
    final encoded = utf8.encode(text);
    final bytes = Uint8List(encoded.length + 1)
      ..setRange(0, encoded.length, encoded);
    if (joinLines) {
      for (var i = 0; i < encoded.length; i++) {
        if (bytes[i] == 0x0a || bytes[i] == 0x0d) {
          bytes[i] = 0x20;
        }
      }
    }

    bytes[encoded.length] = 0x0a; // the trailing \n yields the </s> token
    return bytes;
  }

  /// Adds one token's rows to [indices], and its hash to [wordHashes] when
  /// the model chains words into n-grams.
  ///
  /// Only tokens that count as words are hashed: a label in the text being
  /// predicted takes no part in either, whether the dictionary knows it or
  /// not.
  void _addToken(
    List<int> indices,
    Uint8List data,
    int start,
    int end,
    List<int>? wordHashes,
  ) {
    final wordId = _index[String.fromCharCodes(data, start, end)];
    if (wordId == null) {
      // Unknown word: labels appearing in the text being predicted are
      // ignored, everything else is broken into character n-grams.
      if (_hasLabelPrefix(data, start, end)) {
        return;
      }

      wordHashes?.add(hash(data, start, end));
      // The end-of-sentence token is the one word fastText never takes the
      // subwords of, even when the dictionary has lost it.
      if (!_equals(data, start, end, _eosBytes)) {
        final wrapped = _wrap(data, start, end);
        _computeSubwords(wrapped, 0, end - start + 2, indices);
      }

      return;
    }

    if (_types[wordId] == _typeLabel) {
      return;
    }

    wordHashes?.add(hash(data, start, end));
    if (_args.maxCharNgram <= 0) {
      indices.add(wordId);
    } else {
      indices.addAll(_subwordsOf(wordId));
    }
  }

  static bool _hasLabelPrefix(Uint8List data, int start, int end) {
    if (end - start < labelPrefix.length) {
      return false;
    }

    for (var i = 0; i < labelPrefix.length; i++) {
      if (data[start + i] != labelPrefix.codeUnitAt(i)) {
        return false;
      }
    }
    return true;
  }

  /// Wraps a word in `<` and `>` — begin and end markers, so that n-grams at
  /// the edges of a word differ from the same n-grams in the middle.
  Uint8List _wrap(Uint8List data, int start, int end) {
    final length = end - start;
    if (_wrapBuffer.length < length + 2) {
      _wrapBuffer = Uint8List(length + 2);
    }
    _wrapBuffer[0] = _bow;
    _wrapBuffer.setRange(1, length + 1, data, start);
    _wrapBuffer[length + 1] = _eow;
    return _wrapBuffer;
  }

  List<int> _subwordsOf(int wordId) {
    final cached = _subwordCache[wordId];
    if (cached != null) {
      return cached;
    }

    final subwords = <int>[wordId];
    if (!_equals(_words[wordId], 0, _words[wordId].length, _eosBytes)) {
      final word = _words[wordId];
      final wrapped = _wrap(word, 0, word.length);
      _computeSubwords(wrapped, 0, word.length + 2, subwords);
    }

    _subwordCache[wordId] = subwords;

    return subwords;
  }

  /// Character n-grams of length `minCharNgram` to `maxCharNgram`. Length is
  /// counted in UTF-8 characters rather than bytes: continuation bytes of
  /// multi-byte sequences (`10xxxxxx`) are skipped.
  void _computeSubwords(Uint8List word, int start, int end, List<int> out) {
    // A model trained without an n-gram table has nowhere to put them.
    if (_args.bucket <= 0) {
      return;
    }
    for (var i = start; i < end; i++) {
      if ((word[i] & 0xC0) == 0x80) {
        continue;
      }

      var j = i;
      for (var n = 1; j < end && n <= _args.maxCharNgram; n++) {
        j++;

        while (j < end && (word[j] & 0xC0) == 0x80) {
          j++;
        }

        if (n >= _args.minCharNgram && !(n == 1 && (i == start || j == end))) {
          _pushHash(out, hash(word, i, j) % _args.bucket);
        }
      }
    }
  }

  /// Bucket index to input-matrix row. A pruned model drops some buckets
  /// outright: those n-grams simply take no part in the prediction.
  void _pushHash(List<int> out, int bucketId) {
    if (_pruneIndexSize == 0 || bucketId < 0) {
      return;
    }

    var row = bucketId;
    if (_pruneIndexSize > 0) {
      final mapped = _pruneIndex[bucketId];
      if (mapped == null) {
        return;
      }

      row = mapped;
    }
    out.add(wordCount + row);
  }
}

/// A 64-bit unsigned value held as two 32-bit halves.
///
/// `addWordNgrams` chains its hashes in `uint64_t`: the arithmetic wraps at
/// 2^64, and each 32-bit hash is widened through a signed `int32_t`, so one
/// with the high bit set arrives with its whole top half set. Both details
/// change the answer.
///
/// On the web an `int` is a double and holds neither, so every step here
/// works on 16-bit pieces that stay inside the 53 bits a mantissa represents
/// exactly. Shifts and masks are avoided above 32 bits for the same reason:
/// dart2js truncates their operands to 32.
class _Uint64 {
  static const _twoTo16 = 0x10000;
  static const _twoTo32 = 0x100000000;
  static const _signBit = 0x80000000;
  static const _allOnes = 0xFFFFFFFF;

  var _high = 0;
  var _low = 0;

  /// Loads a 32-bit hash the way C++ widens it, through `int32_t`.
  void setSignExtended(int hash) {
    _low = hash;
    _high = hash >= _signBit ? _allOnes : 0;
  }

  /// `value = value * multiplier + addend`, wrapping at 2^64, with [addend]
  /// widened the same way as in [setSignExtended].
  void multiplyAdd(int multiplier, int addend) {
    final lowPiece = _low % _twoTo16 * multiplier;
    final highPiece = _low ~/ _twoTo16 * multiplier;
    final low = highPiece % _twoTo16 * _twoTo16 + lowPiece;
    final carried = highPiece ~/ _twoTo16 + low ~/ _twoTo32;
    final high =
        _high ~/ _twoTo16 * multiplier % _twoTo16 * _twoTo16 +
        _high % _twoTo16 * multiplier +
        carried;

    final sum = low % _twoTo32 + addend;
    _low = sum % _twoTo32;
    _high =
        (high + sum ~/ _twoTo32 + (addend >= _signBit ? _allOnes : 0)) %
        _twoTo32;
  }

  /// The value modulo [divisor], folded in from the top so that nothing
  /// intermediate needs more than 47 bits.
  int modulo(int divisor) {
    var rest = _high ~/ _twoTo16 % divisor;
    rest = (rest * _twoTo16 + _high % _twoTo16) % divisor;
    rest = (rest * _twoTo16 + _low ~/ _twoTo16) % divisor;

    return (rest * _twoTo16 + _low % _twoTo16) % divisor;
  }
}
