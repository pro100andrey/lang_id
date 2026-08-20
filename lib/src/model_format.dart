/// Constants of the fastText model file format, shared by the reader and the
/// downloader.
library;

/// `FASTTEXT_FILEFORMAT_MAGIC_INT32` — the first four bytes of every model.
const fastTextMagic = 793712314;

/// The newest format version this package understands.
const fastTextSupportedVersion = 12;
