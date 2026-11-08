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
  ) : _subwordCache = List<Int32List?>.filled(wordCount, null),
      _eosId = _index.lookup(_eosBytes, 0, _eosBytes.length);

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
    for (var i = 0; i < size; i++) {
      // Copied, not viewed. A view keeps the whole source buffer alive for
      // as long as the dictionary lives, and for a dense model that buffer
      // is 125 MB that nothing else needs once the matrix has been read.
      // The words themselves are a few hundred kilobytes.
      final word = Uint8List.fromList(reader.cString());
      words.add(word);
      counts[i] = reader.int64();
      types[i] = reader.uint8(); // entry_type is exactly one byte
    }

    final index = _WordTable(words);
    final pruneIndex = _IntTable(pruneIndexSize < 0 ? 0 : pruneIndexSize);
    for (var i = 0; i < pruneIndexSize; i++) {
      final from = reader.int32();
      final to = reader.int32();
      // A negative row would land among the words' own rows, quietly
      // scoring the wrong vector.
      if (to < 0) {
        throw FormatException(
          'the prune index maps bucket $from to row $to',
        );
      }
      // A negative bucket is never looked up: hashes are taken modulo the
      // table size.
      if (from >= 0) {
        pruneIndex.put(from, to);
      }
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

  /// Word to index, looked up by the raw bytes — never decoded, since
  /// malformed UTF-8 would collapse into U+FFFD and produce false matches.
  final _WordTable _index;

  /// The prune map size as stored in the file: -1 means the model is not
  /// pruned, 0 means it is pruned down to nothing (every character n-gram is
  /// dropped), and a positive value is the size of the remapping.
  final int _pruneIndexSize;

  /// Remapping of n-gram buckets after pruning (quantized models only).
  final _IntTable _pruneIndex;

  final List<Int32List?> _subwordCache;

  /// Where the end-of-sentence token is in the dictionary, or -1.
  final int _eosId;
  var _wrapBuffer = Uint8List(_initialWrapBuffer);

  static const _initialWrapBuffer = 64;

  /// The longest word the shared scratch buffer is kept for.
  ///
  /// Words are short: the longest anyone writes runs to a few dozen
  /// characters. A token, though, is whatever sits between two spaces, and a
  /// base64 blob or a line of minified JavaScript arrives as one of any size.
  /// Growing the buffer to fit it used to leave it that size for the life of
  /// the dictionary, with nothing to release it — and the README asks for one
  /// long-lived identifier per isolate, so a server reading text it did not
  /// write got a memory floor chosen by whoever sent the text.
  static const _maxWrapBuffer = 1024;

  /// Size of the reusable buffer. Exposed so that a test can hold the
  /// dictionary to the promise above.
  int get scratchSize => _wrapBuffer.length;

  /// `true` when the dictionary has been pruned, as in `lid.176.ftz`.
  bool get isPruned => _pruneIndexSize >= 0;

  /// The fewest input-matrix rows that every index [lineToIndices] can
  /// produce stays inside of.
  ///
  /// The words take the first [wordCount] rows. A model that hashes
  /// n-grams adds the hash table after them: all of it when unpruned, and
  /// up to the highest row the prune index maps to when pruned.
  int get inputRowsNeeded {
    final hashesNgrams = _args.maxCharNgram > 0 || _args.wordNgrams > 1;
    if (!hashesNgrams || _pruneIndexSize == 0) {
      return wordCount;
    }

    if (_pruneIndexSize < 0) {
      return wordCount + _args.bucket;
    }

    return wordCount + _pruneIndex.highestValue + 1;
  }

  /// Whether [indices], as [lineToIndices] returned them, hold anything but
  /// the end-of-sentence token that fastText appends to every line.
  bool hasContent(List<int> indices) {
    for (final index in indices) {
      if (index != _eosId) {
        return true;
      }
    }

    return false;
  }

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
    var h = _fnvOffset;
    for (var i = start; i < end; i++) {
      h = _fnvStep(h, data[i]);
    }

    return h;
  }

  static const _fnvOffset = 2166136261;
  static const _fnvPrime = 16777619;

  /// One byte of FNV-1a.
  @pragma('vm:prefer-inline')
  static int _fnvStep(int h, int byte) {
    final mixed = (h ^ (byte <= 127 ? byte : 0xFFFFFF00 | byte)) & 0xFFFFFFFF;

    return _numbersAreDoubles
        ? _multiplyMod32(mixed, _fnvPrime)
        : (mixed * _fnvPrime) & 0xFFFFFFFF;
  }

  /// `true` on the web, where `int` is a double. A constant, so each build
  /// keeps only its own branch of [_fnvStep].
  static const bool _numbersAreDoubles = identical(1, 1.0);

  /// `a * b mod 2^32`. On the web the product of two 32-bit numbers does
  /// not fit in a 53-bit mantissa, so multiply by halves. On the VM it fits
  /// in 64 bits — the prime is 25 bits wide — and [_fnvStep] multiplies
  /// directly.
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
    // No guard on the table size: a model that asks for word n-grams without
    // a table to put them in contradicts itself, and ModelArgs.read refuses
    // it before anything gets this far.
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
    // A miss is the common case for lid.176, whose dictionary is small and
    // leans on n-grams; looking up by bytes spares a string per token.
    final wordId = _index.lookup(data, start, end);
    if (wordId < 0) {
      // Unknown word: labels appearing in the text being predicted are
      // ignored, everything else is broken into character n-grams.
      if (_hasLabelPrefix(data, start, end)) {
        return;
      }

      wordHashes?.add(hash(data, start, end));
      // The end-of-sentence token is the one word fastText never takes the
      // subwords of, even when the dictionary has lost it. A model without
      // character n-grams has none to take, and the copy into the scratch
      // buffer would be wasted on every unknown word.
      if (_args.maxCharNgram > 0 && !_equals(data, start, end, _eosBytes)) {
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
    final buffer = _scratchFor(length + 2);
    buffer[0] = _bow;
    buffer.setRange(1, length + 1, data, start);
    buffer[length + 1] = _eow;

    return buffer;
  }

  /// A buffer of at least [size] bytes.
  ///
  /// Ordinary words share one that grows to fit them; anything past
  /// [_maxWrapBuffer] gets a buffer of its own, which is rubbish the moment
  /// its n-grams have been taken.
  Uint8List _scratchFor(int size) {
    if (size > _maxWrapBuffer) {
      return Uint8List(size);
    }

    if (_wrapBuffer.length < size) {
      _wrapBuffer = Uint8List(size);
    }

    return _wrapBuffer;
  }

  Int32List _subwordsOf(int wordId) {
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

    // Packed: a growable list holds twice the room it needs, at twice the
    // width, and the cache lives as long as the model.
    final packed = Int32List.fromList(subwords);
    _subwordCache[wordId] = packed;

    return packed;
  }

  /// Character n-grams of length `minCharNgram` to `maxCharNgram`. Length is
  /// counted in UTF-8 characters rather than bytes: continuation bytes of
  /// multi-byte sequences (`10xxxxxx`) are skipped.
  ///
  /// There is no guard on the table size: ModelArgs.read refuses a model
  /// that asks for character n-grams without a table to hash them into.
  void _computeSubwords(Uint8List word, int start, int end, List<int> out) {
    for (var i = start; i < end; i++) {
      if ((word[i] & 0xC0) == 0x80) {
        continue;
      }

      // FNV-1a folds one byte at a time, so the hash of each n-gram from i
      // carries on from the one before it rather than starting over.
      var h = _fnvOffset;
      var j = i;
      for (var n = 1; j < end && n <= _args.maxCharNgram; n++) {
        h = _fnvStep(h, word[j++]);

        while (j < end && (word[j] & 0xC0) == 0x80) {
          h = _fnvStep(h, word[j++]);
        }

        if (n >= _args.minCharNgram && !(n == 1 && (i == start || j == end))) {
          _pushHash(out, h % _args.bucket);
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
      final mapped = _pruneIndex.get(bucketId);
      if (mapped < 0) {
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

/// Word bytes to dictionary index: open addressing over the FNV hash, the
/// way fastText's own `word2int_` works, so a lookup needs no string.
class _WordTable {
  _WordTable(this._words) : _slots = _emptySlots(_capacityFor(_words.length)) {
    for (var id = 0; id < _words.length; id++) {
      _insert(id);
    }
  }

  final List<Uint8List> _words;
  final Int32List _slots;

  static Int32List _emptySlots(int capacity) =>
      Int32List(capacity)..fillRange(0, capacity, -1);

  /// A power of two at least twice [entries], so probes stay short.
  static int _capacityFor(int entries) {
    var capacity = 16;
    while (capacity < entries * 2) {
      capacity *= 2;
    }

    return capacity;
  }

  /// Adds the word at [id]. A word seen before takes the later index, as a
  /// map assignment would.
  void _insert(int id) {
    final word = _words[id];
    final mask = _slots.length - 1;
    var slot = Dictionary.hash(word, 0, word.length) & mask;
    while (true) {
      final existing = _slots[slot];
      if (existing < 0 || _sameBytes(word, 0, word.length, _words[existing])) {
        _slots[slot] = id;

        return;
      }
      slot = (slot + 1) & mask;
    }
  }

  /// The index of the word in `data[start, end)`, or -1.
  int lookup(Uint8List data, int start, int end) {
    final mask = _slots.length - 1;
    var slot = Dictionary.hash(data, start, end) & mask;
    while (true) {
      final id = _slots[slot];
      if (id < 0 || _sameBytes(data, start, end, _words[id])) {
        return id;
      }
      slot = (slot + 1) & mask;
    }
  }

  static bool _sameBytes(Uint8List data, int start, int end, Uint8List word) {
    if (end - start != word.length) {
      return false;
    }

    for (var i = 0; i < word.length; i++) {
      if (data[start + i] != word[i]) {
        return false;
      }
    }

    return true;
  }
}

/// A map from non-negative int32 keys to non-negative int32 values, in two
/// flat lists. The prune index of `lid.176.ftz` has 42,765 entries and is
/// read for every n-gram of every text.
class _IntTable {
  _IntTable(int entries)
    : _keys = _WordTable._emptySlots(_WordTable._capacityFor(entries)),
      _values = Int32List(_WordTable._capacityFor(entries));

  final Int32List _keys;
  final Int32List _values;

  /// The largest value stored, or -1 when there is none.
  var highestValue = -1;

  void put(int key, int value) {
    final mask = _keys.length - 1;
    // The keys are bucket ids, themselves the output of a hash, so their
    // low bits are already spread.
    var slot = key & mask;
    while (_keys[slot] >= 0 && _keys[slot] != key) {
      slot = (slot + 1) & mask;
    }
    _keys[slot] = key;
    _values[slot] = value;
    if (value > highestValue) {
      highestValue = value;
    }
  }

  /// The value stored for [key], or -1.
  int get(int key) {
    final mask = _keys.length - 1;
    var slot = key & mask;
    while (true) {
      final stored = _keys[slot];
      if (stored == key) {
        return _values[slot];
      }
      if (stored < 0) {
        return -1;
      }
      slot = (slot + 1) & mask;
    }
  }
}
