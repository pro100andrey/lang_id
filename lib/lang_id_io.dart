/// Loading a model from disk. Kept in its own library so that
/// `package:lang_id/lang_id.dart` stays usable on the web, where there is
/// no `dart:io`.
library;

import 'dart:io';

import 'src/language_identifier.dart';
import 'src/model_downloader.dart';

export 'lang_id.dart';
export 'src/model_downloader.dart'
    show
        DownloadProgress,
        ModelDownloadException,
        ModelDownloader,
        PretrainedModel;

/// Reads a model from a `.bin` or `.ftz` file.
Future<LanguageIdentifier> loadModel(String path) async =>
    LanguageIdentifier.fromBytes(await File(path).readAsBytes());

/// Synchronous variant of [loadModel], for CLIs and tests.
LanguageIdentifier loadModelSync(String path) =>
    LanguageIdentifier.fromBytes(File(path).readAsBytesSync());

/// Loads [model] from [directory], downloading it first if it is not there.
///
/// The one-liner most callers want: the weights are not shipped with the
/// package, so the first run fetches them and every later run reads the file
/// that is already on disk.
///
/// ```dart
/// final identifier = await loadOrDownloadModel(
///   PretrainedModel.compact,
///   directory: 'models',
/// );
/// ```
///
/// Pass [downloader] to control the mirror, the [HttpClient] or the retry
/// policy; otherwise a default one is created and closed here.
Future<LanguageIdentifier> loadOrDownloadModel(
  PretrainedModel model, {
  required String directory,
  void Function(DownloadProgress progress)? onProgress,
  ModelDownloader? downloader,
}) async {
  final client = downloader ?? ModelDownloader();
  try {
    final file = await client.download(
      model,
      directory: directory,
      onProgress: onProgress,
    );
    return .fromBytes(await file.readAsBytes());
  } finally {
    if (downloader == null) {
      await client.close();
    }
  }
}
