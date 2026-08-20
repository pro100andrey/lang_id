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
  final directory = _optionValue(args, '--out') ?? 'models';
  final force = args.contains('--force');

  final downloader = ModelDownloader();
  final target = downloader.fileIn(directory, model);
  if (target.existsSync() && !force) {
    stdout.writeln(
      '${target.path} is already here (${_humanSize(target.lengthSync())})',
    );
    await downloader.close();
    return;
  }

  stdout.writeln('Downloading ${downloader.urlOf(model)}');
  try {
    final file = await downloader.download(
      model,
      directory: directory,
      force: force,
      onProgress: _report,
    );
    stdout
      ..writeln('\nDone: ${file.path}')
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

String? _optionValue(List<String> args, String name) {
  final index = args.indexOf(name);
  if (index == -1 || index + 1 >= args.length) {
    return null;
  }
  return args[index + 1];
}

String _humanSize(int bytes) => bytes >= 1024 * 1024
    ? '${(bytes / 1024 / 1024).toStringAsFixed(1)} MB'
    : '${(bytes / 1024).toStringAsFixed(0)} KB';
