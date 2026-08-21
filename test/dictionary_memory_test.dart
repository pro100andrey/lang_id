import 'package:lang_id/lang_id.dart';
import 'package:lang_id/src/binary_reader.dart';
import 'package:lang_id/src/dictionary.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// What a loaded model is allowed to hold on to.
///
/// Both of these are about a cost that outlives the call that caused it,
/// which is the kind a long-lived identifier turns into a leak: the README
/// asks for one per isolate, so anything it keeps, it keeps for good.
void main() {
  Dictionary readDictionary() {
    final reader = BinaryReader(buildSyntheticModel())
      ..int32()
      ..int32();

    return Dictionary.read(reader, ModelArgs.read(reader, 12));
  }

  test('an enormous token does not become a permanent cost', () {
    final dictionary = readDictionary();
    final before = dictionary.scratchSize;

    // One token of a megabyte: a base64 blob, or a line of minified
    // JavaScript. The buffer it needs used to replace the small shared one
    // and stay for the life of the dictionary.
    dictionary.lineToIndices('x' * (1 << 20));

    expect(dictionary.scratchSize, before);
    expect(before, lessThanOrEqualTo(1024));
  });

  test('ordinary words still share one buffer', () {
    final dictionary = readDictionary()
      ..lineToIndices('unbekannte woerter hier entlang');

    expect(dictionary.scratchSize, lessThanOrEqualTo(1024));
  });

  test('a dense model reads nothing from the source bytes after loading', () {
    // The dictionary copies its words rather than viewing the buffer, so
    // nothing of a dense model's file survives the load — which is what
    // keeps a 125 MB .bin from costing 250. Overwriting every byte of the
    // source cannot change an answer.
    final bytes = buildSyntheticModel();
    final identifier = LanguageIdentifier.fromBytes(bytes);
    final before = identifier.predict('alpha', k: 2);

    bytes.fillRange(0, bytes.length, 0);

    expect(identifier.predict('alpha', k: 2), before);
    expect(identifier.labels, ['x', 'y']);
  });
}
