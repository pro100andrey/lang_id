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
/// be a model is fetched again. Pass [force] to fetch even when the file is
/// fine.
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
    final file = await client.download(
      model,
      directory: directory,
      force: force,
      onProgress: onProgress,
    );
    return .fromBytes(await file.readAsBytes());
  } finally {
    if (downloader == null) {
      await client.close();
    }
  }
}
