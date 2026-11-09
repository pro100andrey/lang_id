/// Loading a model from disk. Kept in its own library so that
/// `package:lang_id/lang_id.dart` stays usable on the web, where there is
/// no `dart:io`.
library;

import 'dart:io';

import 'src/fasttext_classifier.dart';
import 'src/model_downloader.dart';

export 'lang_id.dart';
export 'src/model_downloader.dart'
    show
        DownloadProgress,
        ModelDownloadException,
        ModelDownloader,
        PretrainedModel;

/// Reads a model from a `.bin` or `.ftz` file.
Future<FastTextClassifier> loadModel(String path) async =>
    FastTextClassifier.fromBytes(await File(path).readAsBytes());

/// Synchronous variant of [loadModel], for CLIs and tests.
FastTextClassifier loadModelSync(String path) =>
    FastTextClassifier.fromBytes(File(path).readAsBytesSync());

/// Loads [model] from [directory], downloading it first if it is not there.
///
/// The one-liner most callers want: the weights are not shipped with the
/// package, so the first run fetches them and every later run reads the file
/// that is already on disk.
///
/// ```dart
/// final classifier = await loadOrDownloadModel(
///   PretrainedModel.compact,
///   directory: 'models',
/// );
/// ```
///
/// A file already in [directory] is read as it is; one that turns out not to
/// be a model, or not a whole one, is fetched again, once. Pass [force] to
/// fetch even when the file is fine.
///
/// Pass [downloader] to control the mirror, the [HttpClient] or the
/// timeouts; otherwise a default one is created and closed here.
Future<FastTextClassifier> loadOrDownloadModel(
  PretrainedModel model, {
  required String directory,
  bool force = false,
  void Function(DownloadProgress progress)? onProgress,
  ModelDownloader? downloader,
}) async {
  final client = downloader ?? ModelDownloader();
  try {
    final before = client.fileIn(directory, model).statSync();
    final file = await client.download(
      model,
      directory: directory,
      force: force,
      onProgress: onProgress,
    );
    try {
      return .fromBytes(await file.readAsBytes());
    } on FormatException {
      // download only checks the header of a file that is already there,
      // and a body cut short behind a good header — what 0.1.0 could cache —
      // fails only here. Fetching once more heals it. A file download just
      // wrote is a real problem and is reported as one: that includes a
      // cached file download itself threw away and fetched again, which is
      // why this compares the file rather than asking whether one existed.
      if (!_servedFromDisk(before, file.statSync())) {
        rethrow;
      }
    }

    final refetched = await client.download(
      model,
      directory: directory,
      force: true,
      onProgress: onProgress,
    );
    return .fromBytes(await refetched.readAsBytes());
  } finally {
    if (downloader == null) {
      await client.close();
    }
  }
}

/// Whether [after] is the very file [before] described: download returned
/// what was already on disk instead of writing a new one. A download renames
/// a freshly written file into place, which changes its modification time.
bool _servedFromDisk(FileStat before, FileStat after) =>
    before.type != FileSystemEntityType.notFound &&
    before.modified == after.modified &&
    before.size == after.size;
