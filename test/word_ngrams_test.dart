import 'package:lang_id/src/binary_reader.dart';
import 'package:lang_id/src/dictionary.dart';
import 'package:lang_id/src/model_args.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// Word n-grams, checked against an independent implementation.
///
/// The rows the reference produces are pinned by test/fixture_parity_test,
/// which runs against models trained by the original fastText. What that
/// cannot do is run on the web, and the chain here is exactly the arithmetic
/// the web breaks: it wraps at 2^64, which a dart2js `int` cannot hold, and
/// it sign-extends each hash through `int32_t`. So the same chain is
/// computed with [BigInt] — slow, obvious and impossible to get subtly wrong
/// — and the two are compared.
void main() {
  const bucket = 100;
  const wordCount = 3;

  Dictionary dictionaryOf({int wordNgrams = 2}) {
    final reader =
        BinaryReader(
            buildSyntheticModel(wordNgrams: wordNgrams, bucket: bucket),
          )
          ..int32()
          ..int32();

    return Dictionary.read(reader, ModelArgs.read(reader, 12));
  }

  int hashOf(String word) {
    final bytes = utf8Bytes(word);

    return Dictionary.hash(bytes, 0, bytes.length);
  }

  group('addWordNgrams', () {
    test('chains neighbouring words the way fastText does', () {
      // alpha and beta are words 1 and 2; the trailing newline yields </s>,
      // which is word 0 and counts as a word, so it takes part in the last
      // n-gram too.
      final indices = dictionaryOf().lineToIndices('alpha beta');

      expect(indices, [
        1,
        2,
        0,
        wordCount + chainedBucket([hashOf('alpha'), hashOf('beta')], bucket),
        wordCount + chainedBucket([hashOf('beta'), hashOf('</s>')], bucket),
      ]);
    });

    test('a longer reach carries between the steps correctly', () {
      // Three words and the end-of-sentence token, chained up to three at a
      // time: three pairs and two triples, in the order fastText emits them.
      // The triples are the ones that matter — they are two multiplications
      // deep, so a carry lost on its way into the second one shows up here
      // and nowhere else.
      final alpha = hashOf('alpha');
      final beta = hashOf('beta');
      final eos = hashOf('</s>');

      expect(dictionaryOf(wordNgrams: 3).lineToIndices('alpha beta alpha'), [
        1,
        2,
        1,
        0,
        wordCount + chainedBucket([alpha, beta], bucket),
        wordCount + chainedBucket([alpha, beta, alpha], bucket),
        wordCount + chainedBucket([beta, alpha], bucket),
        wordCount + chainedBucket([beta, alpha, eos], bucket),
        wordCount + chainedBucket([alpha, eos], bucket),
      ]);
    });

    test('reaches only as far as wordNgrams allows', () {
      // Three words and a trailing </s> make three pairs at n = 2, and three
      // pairs plus two triples at n = 3.
      expect(
        dictionaryOf().lineToIndices('alpha beta alpha'),
        hasLength(4 + 3),
      );
      expect(
        dictionaryOf(wordNgrams: 3).lineToIndices('alpha beta alpha'),
        hasLength(4 + 3 + 2),
      );
    });

    test('a model without them is untouched', () {
      expect(dictionaryOf(wordNgrams: 1).lineToIndices('alpha beta'), [
        1,
        2,
        0,
      ]);
    });

    test('a label in the text takes part in nothing', () {
      // fastText ignores labels while predicting, and they are not hashed
      // either, so the n-gram jumps across them.
      expect(
        dictionaryOf().lineToIndices('alpha __label__x beta'),
        dictionaryOf().lineToIndices('alpha beta'),
      );
    });
  });
}

/// The bucket fastText's `addWordNgrams` puts a run of word hashes in.
///
/// `h = h * 116049371 + next` in `uint64_t` for every hash after the first,
/// each one widened from `int32_t`, which sign-extends the ones with the
/// high bit set.
///
/// Taking a whole run rather than a pair is the point: one step of the
/// chunked arithmetic can be right while the carry it hands to the next one
/// is not, and only a run longer than two shows that.
int chainedBucket(List<int> hashes, int bucket) {
  final wrap = BigInt.two.pow(64) - BigInt.one;
  var chained = signExtended(hashes.first);
  for (final hash in hashes.skip(1)) {
    chained = (chained * BigInt.from(116049371) + signExtended(hash)) & wrap;
  }

  return (chained % BigInt.from(bucket)).toInt();
}

BigInt signExtended(int hash) => hash >= 0x80000000
    ? BigInt.from(hash) | BigInt.parse('ffffffff00000000', radix: 16)
    : BigInt.from(hash);
