/// Constants of the fastText model file format, shared by the reader and the
/// downloader.
library;

/// `FASTTEXT_FILEFORMAT_MAGIC_INT32` — the first four bytes of every model.
const fastTextMagic = 793712314;

/// The newest format version this package understands.
const fastTextSupportedVersion = 12;

/// Checks the two fields every model file begins with.
///
/// Shared so that the reader and the downloader cannot drift apart: they
/// used to carry a copy each, over different readers, raising different
/// exceptions for the same bytes.
void checkFastTextHeader(int magic, int version) {
  if (magic != fastTextMagic) {
    throw const FormatException('not a fastText model: wrong file signature');
  }

  if (version > fastTextSupportedVersion) {
    throw FormatException(
      'format version $version is newer than the supported one '
      '($fastTextSupportedVersion)',
    );
  }
}
