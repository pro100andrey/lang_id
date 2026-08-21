import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'fasttext_classifier.dart';
import 'model_format.dart';

/// A pretrained language identification model published by Facebook.
///
/// Both variants recognize the same 176 languages. The weights are **not**
/// shipped with this package: they are distributed under CC BY-SA 3.0 and are
/// fetched at run time by [ModelDownloader]. If you redistribute the file, see
/// `NOTICE.md` for what the licence asks of you.
enum PretrainedModel {
  /// `lid.176.ftz`, about 916 KB — compressed with product quantization.
  /// Gives up almost nothing on ordinary text; the sensible default.
  compact('lid.176.ftz', 938013),

  /// `lid.176.bin`, about 125 MB — the full dense model.
  full('lid.176.bin', 131266198);

  const PretrainedModel(this.fileName, this.publishedSize);

  /// Name of the published file, and of the file written to disk.
  final String fileName;

  /// Size of the published file as of writing.
  ///
  /// A number to show before a download starts — how much a first run is
  /// about to cost. Nothing here measures against it: a mirror may serve a
  /// different build, and progress is reported from what the server actually
  /// announces.
  final int publishedSize;
}

/// How far a download has got.
class DownloadProgress {
  const DownloadProgress(this.model, this.receivedBytes, this.totalBytes);

  /// Which model is being fetched.
  final PretrainedModel model;

  /// Bytes written so far.
  final int receivedBytes;

  /// Total size reported by the server, or `null` when it did not say.
  final int? totalBytes;

  /// Completed fraction between 0 and 1, or `null` when the total is unknown.
  double? get fraction {
    final total = totalBytes;
    if (total == null || total <= 0) {
      return null;
    }

    return receivedBytes / total;
  }

  @override
  String toString() {
    final done = fraction;
    final percent = done == null ? '?' : (done * 100).toStringAsFixed(1);

    return '${model.fileName} $percent% ($receivedBytes bytes)';
  }
}

/// Thrown when a model cannot be fetched or what arrived is not a model.
class ModelDownloadException implements Exception {
  const ModelDownloadException(this.message, {this.uri, this.statusCode});

  /// What went wrong, in one line.
  final String message;

  /// The address that was being fetched, when there was one.
  final Uri? uri;

  /// The HTTP status, when the server answered at all.
  final int? statusCode;

  @override
  String toString() {
    final where = uri == null ? '' : ' ($uri)';
    final status = statusCode == null ? '' : ' [HTTP $statusCode]';
    return 'ModelDownloadException: $message$where$status';
  }
}

/// Fetches pretrained models and puts them in a directory of your choosing.
///
/// ```dart
/// final downloader = ModelDownloader();
/// try {
///   final file = await downloader.download(
///     PretrainedModel.compact,
///     directory: 'assets/models',
///     onProgress: (p) => print(p),
///   );
///   final classifier = FastTextClassifier.fromBytes(await file.readAsBytes());
/// } finally {
///   await downloader.close();
/// }
/// ```
///
/// A model already present in the directory is returned as is, unless it
/// turns out not to be a model, in which case it is thrown away and fetched
/// again; pass `force: true` to refetch regardless. The download goes to a
/// `.part` file and is renamed into place only once it is complete and checks
/// out, so an interrupted run never leaves a half-written model behind.
class ModelDownloader {
  /// Creates a downloader.
  ///
  /// [baseUrl] lets you point at a mirror or an internal copy; it must end
  /// with a slash for the file name to resolve against it. Pass [httpClient]
  /// to reuse a client you already configured — proxies, certificates,
  /// timeouts; in that case closing it stays your responsibility, and
  /// [connectionTimeout] is left to whatever you set on it.
  ///
  /// [connectionTimeout] bounds reaching the server and getting an answer
  /// out of it, [stallTimeout] the longest silence allowed in the middle of
  /// the body. Without them a server that accepts the connection and then
  /// says nothing holds the caller forever — and this is the call an app
  /// makes while starting up.
  ModelDownloader({
    Uri? baseUrl,
    HttpClient? httpClient,
    this.connectionTimeout = const Duration(seconds: 30),
    this.stallTimeout = const Duration(seconds: 60),
  }) : baseUrl = baseUrl ?? defaultBaseUrl,
       _httpClient = httpClient ?? HttpClient(),
       _ownsClient = httpClient == null {
    if (_ownsClient) {
      _httpClient.connectionTimeout = connectionTimeout;
    }
  }

  /// Where Facebook publishes the models.
  static final Uri defaultBaseUrl = Uri.parse(
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/',
  );

  /// Where models are fetched from; ends with a slash.
  final Uri baseUrl;

  /// How long the server has to answer at all.
  final Duration connectionTimeout;

  /// How long the body may go quiet before the download is abandoned.
  final Duration stallTimeout;

  final HttpClient _httpClient;
  final bool _ownsClient;
  var _closed = false;

  /// The address [download] will fetch [model] from.
  Uri urlOf(PretrainedModel model) => baseUrl.resolve(model.fileName);

  /// Where [download] will put [model] inside [directory].
  ///
  /// Useful for checking whether the model is already in place without
  /// touching the network.
  ///
  /// The name is joined on rather than resolved as a URI, which would
  /// normalize `..` by text — `a/../b` is `b` only when `a` is not a symlink,
  /// and the kernel that later creates the directory knows that and a URI
  /// does not. An empty [directory] means the working directory.
  File fileIn(String directory, PretrainedModel model) {
    if (directory.isEmpty) {
      return File(model.fileName);
    }

    final separator =
        directory.endsWith(Platform.pathSeparator) || directory.endsWith('/')
        ? ''
        : Platform.pathSeparator;

    return File('$directory$separator${model.fileName}');
  }

  /// Downloads [model] into [directory], creating the directory if needed.
  ///
  /// Returns the file on disk. When it is already there and [force] is false,
  /// nothing is fetched — unless what is there is not a model, which is
  /// thrown away and fetched again rather than reported forever.
  /// [onProgress] is called as bytes arrive.
  ///
  /// Throws [ModelDownloadException] when the server refuses, the connection
  /// fails or stalls, or the bytes that arrive are not a whole fastText
  /// model.
  Future<File> download(
    PretrainedModel model, {
    required String directory,
    bool force = false,
    void Function(DownloadProgress progress)? onProgress,
  }) async {
    if (_closed) {
      throw StateError('this ModelDownloader has been closed');
    }

    final url = urlOf(model);
    final target = fileIn(directory, model);
    try {
      if (!force && target.existsSync()) {
        if (await _looksLikeModel(target)) {
          return target;
        }

        // Whatever is there is not a model — an error page saved under the
        // right name, or the tail of a download that was cut short. Reporting
        // that on every call for the rest of time only leaves the caller with
        // a file to delete by hand.
        await target.delete();
      }
    } on FileSystemException catch (error) {
      throw ModelDownloadException(
        'could not look at ${target.path}: $error',
        uri: url,
      );
    }

    await Directory(directory).create(recursive: true);
    await _discardStaleParts(target);
    // A name of its own per attempt. Two isolates fetching the same model
    // used to open the same scratch file, interleave their chunks, and
    // rename the mixture into place, where it passed the header check and
    // was cached.
    final partial = File('${target.path}.$pid.${_attempts++}.part');

    try {
      final request = await _httpClient.getUrl(url).timeout(connectionTimeout);
      final response = await request.close().timeout(connectionTimeout);
      if (response.statusCode != HttpStatus.ok) {
        await response.drain<void>();
        throw ModelDownloadException(
          'server refused to serve the model',
          uri: url,
          statusCode: response.statusCode,
        );
      }

      // A compressed response counts its bytes on the wire, while what
      // arrives here is what they unpack to, so the announced length is not
      // the length of anything this can see: it would put progress at 600%
      // and make the check below reject a download that was fine.
      final encoding = response.headers.value(
        HttpHeaders.contentEncodingHeader,
      );
      final packed = encoding != null && encoding.toLowerCase() != 'identity';
      final total = !packed && response.contentLength >= 0
          ? response.contentLength
          : null;
      var received = 0;
      final sink = partial.openWrite();
      try {
        await for (final chunk in response.timeout(stallTimeout)) {
          sink.add(chunk);
          received += chunk.length;
          onProgress?.call(DownloadProgress(model, received, total));
        }
      } finally {
        await sink.close();
      }

      await _verifyWhole(partial, model, received, total);
      if (target.existsSync()) {
        await target.delete();
      }

      return await partial.rename(target.path);
    } on ModelDownloadException {
      await _discard(partial);
      rethrow;
    } on TimeoutException {
      await _discard(partial);
      throw ModelDownloadException(
        'the server stopped sending the model',
        uri: url,
      );
    } on IOException catch (error, stack) {
      // Only the failures of reaching a server and writing a file are turned
      // into a download exception, and the stack comes with them. Catching
      // everything used to rewrite a bug in the caller's own onProgress
      // callback as a network problem, and throw away where it happened.
      await _discard(partial);
      Error.throwWithStackTrace(
        ModelDownloadException(
          'could not download the model: $error',
          uri: url,
        ),
        stack,
      );
    } on Object {
      await _discard(partial);
      rethrow;
    }
  }

  /// Releases the underlying [HttpClient], unless it was passed in.
  Future<void> close() async {
    _closed = true;
    if (_ownsClient) {
      _httpClient.close();
    }
  }

  /// How many attempts this isolate has made, so that two of them never
  /// pick the same scratch file.
  static var _attempts = 0;

  /// Removes scratch files left by a run that was killed outright.
  ///
  /// Anything younger than this could belong to a download that is still
  /// going on in another process, which is the whole reason the names are
  /// unique; anything older is not coming back, and the full model is
  /// 125 MB to leave lying around.
  static const _staleAfter = Duration(hours: 1);

  static Future<void> _discardStaleParts(File target) async {
    final directory = target.parent;
    if (!directory.existsSync()) {
      return;
    }

    final cutoff = DateTime.now().subtract(_staleAfter);
    final prefix = '${target.path}.';
    for (final entry in directory.listSync()) {
      if (entry is! File ||
          !entry.path.startsWith(prefix) ||
          !entry.path.endsWith('.part')) {
        continue;
      }

      try {
        if (entry.statSync().modified.isBefore(cutoff)) {
          await entry.delete();
        }
      } on FileSystemException {
        // Someone else's leftovers, or someone else got there first.
      }
    }
  }

  static Future<void> _discard(File partial) async {
    if (partial.existsSync()) {
      try {
        await partial.delete();
      } on FileSystemException {
        // Nothing useful to do: the caller already has a real failure.
      }
    }
  }

  /// Whether [file] at least begins like a fastText model.
  ///
  /// Used on a file that is already in place, where reading all of it — 125
  /// MB, for the full model — to answer a question the caller is about to
  /// answer anyway would be wasteful.
  static Future<bool> _looksLikeModel(File file) async {
    try {
      await _verifyHeader(file, null);

      return true;
    } on ModelDownloadException {
      return false;
    }
  }

  /// Checks that what arrived is a whole model.
  ///
  /// The byte count is the cheap half and only works when the server said
  /// how much it was sending. Without a `Content-Length` — a chunked or
  /// close-delimited response — a body cut in half ends the stream just as
  /// normally as a complete one, and the eight bytes of the header are
  /// already there, so the truncated file used to be renamed into place and
  /// cached forever. The only thing left that can tell is reading it.
  static Future<void> _verifyWhole(
    File file,
    PretrainedModel model,
    int received,
    int? total,
  ) async {
    if (total != null) {
      if (received != total) {
        throw ModelDownloadException(
          '${model.fileName} stopped after $received of $total bytes',
        );
      }

      await _verifyHeader(file, model);

      return;
    }

    try {
      FastTextClassifier.fromBytes(await file.readAsBytes());
    } on FormatException catch (error) {
      throw ModelDownloadException(
        '${model.fileName} did not arrive whole: ${error.message}',
      );
    }
  }

  /// Checks the four magic bytes and the format version. An HTML error page
  /// saved as a model fails here rather than much later, inside the reader.
  static Future<void> _verifyHeader(File file, PretrainedModel? model) async {
    final handle = await file.open();
    Uint8List head;
    try {
      head = await handle.read(8);
    } finally {
      await handle.close();
    }

    if (head.length < 8) {
      throw ModelDownloadException(
        '${file.path} is too short to be a fastText model',
      );
    }

    final data = ByteData.view(head.buffer, head.offsetInBytes, head.length);
    try {
      checkFastTextHeader(
        data.getInt32(0, Endian.little),
        data.getInt32(4, Endian.little),
      );
    } on FormatException catch (error) {
      throw ModelDownloadException(
        '${model?.fileName ?? file.path}: ${error.message}',
      );
    }
  }
}
