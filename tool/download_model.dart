// Downloads Facebook's language identification model.
//
//   dart run tool/download_model.dart                    # lid.176.ftz, 916 KB
//   dart run tool/download_model.dart --full             # lid.176.bin, 125 MB
//   dart run tool/download_model.dart --out assets       # somewhere else
//
// A thin shell around ModelDownloader from the package itself, so an app can
// do exactly the same thing at run time without shelling out to this script.
//
// The models are distributed under CC BY-SA 3.0 and are not committed to the
// repository. See NOTICE.md for the licence terms.
import 'dart:io';

import 'package:lang_id/lang_id_io.dart';

Future<void> main(List<String> args) async {
  final model = args.contains('--full')
      ? PretrainedModel.full
      : PretrainedModel.compact;
  final force = args.contains('--force');
  final directory = _optionValue(args, '--out', fallback: 'models');
  if (directory == null) {
    stderr.writeln('--out needs a directory to put the model in');
    exitCode = 2;

    return;
  }

  final downloader = ModelDownloader();
  final target = downloader.fileIn(directory, model);
  final wasHere = target.existsSync() && !force;
  // Whether the file is usable is the downloader's judgement, not ours: it
  // returns straight away when the model is already in place, and refetches
  // when what is in place is an error page saved under the right name. This
  // script used to answer "already here" to that file and exit successfully,
  // while the library refused the very same bytes.
  stdout.writeln(
    wasHere
        ? 'Checking ${target.path}'
        : 'Downloading ${downloader.urlOf(model)}',
  );
  try {
    final file = await downloader.download(
      model,
      directory: directory,
      force: force,
      onProgress: _report,
    );
    stdout
      ..writeln('\nDone: ${file.path} (${_humanSize(file.lengthSync())})')
      ..writeln(
        '\nThe model is distributed under CC BY-SA 3.0, '
        '(c) Facebook, Inc. — see NOTICE.md',
      );
  } on ModelDownloadException catch (error) {
    stderr.writeln('\n$error');
    exitCode = 1;
  } finally {
    await downloader.close();
  }
}

void _report(DownloadProgress progress) {
  final done = progress.fraction;
  final size = _humanSize(progress.receivedBytes);
  if (done == null) {
    stdout.write('\r$size');
    return;
  }
  stdout.write(
    '\r${(done * 100).toStringAsFixed(1)}%  '
    '$size of ${_humanSize(progress.totalBytes!)}',
  );
}

/// The value written after [name], [fallback] when the option is absent, or
/// null when it is there without one.
///
/// The last case used to be read as whatever came next: `--out --force`
/// fetched the model into a directory literally called `--force`, and still
/// forced it.
String? _optionValue(
  List<String> args,
  String name, {
  required String fallback,
}) {
  final index = args.indexOf(name);
  if (index == -1) {
    return fallback;
  }

  if (index + 1 >= args.length || args[index + 1].startsWith('-')) {
    return null;
  }

  return args[index + 1];
}

String _humanSize(int bytes) => bytes >= 1024 * 1024
    ? '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB'
    : '${(bytes / 1024).toStringAsFixed(0)} KB';
