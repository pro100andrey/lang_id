@TestOn('vm')
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:lang_id/lang_id_io.dart';
import 'package:test/test.dart';

import 'support/synthetic_model.dart';

/// The downloader is tested against a local server, so the suite stays
/// offline and deterministic: the served bytes are the synthetic model, which
/// the reader can actually load.
void main() {
  late HttpServer server;
  late Directory workDir;
  late ModelDownloader downloader;

  /// What the next request will be answered with.
  late List<int> body;
  late int status;
  var requestCount = 0;
  var sendContentLength = true;

  /// Stops the response after this many bytes, without saying so — what a
  /// connection cut in the middle looks like when there is no Content-Length.
  int? cutAfter;

  /// Answers the request and then says nothing at all.
  var goQuiet = false;

  /// Compresses the body, the way a CDN does, so that the announced length
  /// is the one on the wire and not the one that arrives.
  var compress = false;

  setUp(() async {
    body = buildSyntheticModel();
    status = HttpStatus.ok;
    requestCount = 0;
    sendContentLength = true;
    cutAfter = null;
    goQuiet = false;
    compress = false;

    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    unawaited(() async {
      await for (final request in server) {
        requestCount++;
        request.response.statusCode = status;
        final served = compress ? gzip.encode(body) : body;
        if (compress) {
          request.response.headers.set(
            HttpHeaders.contentEncodingHeader,
            'gzip',
          );
        }

        if (sendContentLength) {
          request.response.contentLength = served.length;
        }
        if (goQuiet) {
          continue;
        }

        request.response.add(
          cutAfter == null ? served : served.sublist(0, cutAfter),
        );
        try {
          await request.response.close();
        } on HttpException {
          // Closing after fewer bytes than announced is the whole point of
          // the truncation cases; the client is the one that has to notice.
        }
      }
    }());

    workDir = await Directory.systemTemp.createTemp('lang_id_download_test');
    downloader = ModelDownloader(
      baseUrl: Uri.parse('http://${server.address.host}:${server.port}/'),
    );
  });

  tearDown(() async {
    await downloader.close();
    await server.close(force: true);
    if (workDir.existsSync()) {
      await workDir.delete(recursive: true);
    }
  });

  String pathIn(String name) => '${workDir.path}/$name';

  test('downloads the model into the requested directory', () async {
    final nested = pathIn('assets/models');
    final file = await downloader.download(
      PretrainedModel.compact,
      directory: nested,
    );

    expect(file.existsSync(), isTrue);
    expect(file.path, endsWith('lid.176.ftz'));
    expect(file.parent.path, endsWith('models'));
    expect(await file.readAsBytes(), body);
  });

  test('creates missing directories', () async {
    final deep = pathIn('a/b/c');
    expect(Directory(deep).existsSync(), isFalse);
    await downloader.download(PretrainedModel.compact, directory: deep);
    expect(Directory(deep).existsSync(), isTrue);
  });

  test('a model already in place is not fetched again', () async {
    await downloader.download(PretrainedModel.compact, directory: workDir.path);
    expect(requestCount, 1);

    final again = await downloader.download(
      PretrainedModel.compact,
      directory: workDir.path,
    );
    expect(requestCount, 1, reason: 'the second call must not hit the network');
    expect(again.existsSync(), isTrue);
  });

  test('force fetches the model again', () async {
    await downloader.download(PretrainedModel.compact, directory: workDir.path);
    await downloader.download(
      PretrainedModel.compact,
      directory: workDir.path,
      force: true,
    );
    expect(requestCount, 2);
  });

  test('reports progress up to the full size', () async {
    final seen = <DownloadProgress>[];
    await downloader.download(
      PretrainedModel.compact,
      directory: workDir.path,
      onProgress: seen.add,
    );

    expect(seen, isNotEmpty);
    expect(seen.last.receivedBytes, body.length);
    expect(seen.last.totalBytes, body.length);
    expect(seen.last.fraction, 1.0);
    expect(seen.last.model, PretrainedModel.compact);
  });

  test('an unknown total size leaves the fraction null', () async {
    sendContentLength = false;
    final seen = <DownloadProgress>[];
    await downloader.download(
      PretrainedModel.compact,
      directory: workDir.path,
      onProgress: seen.add,
    );

    expect(seen.last.totalBytes, isNull);
    expect(seen.last.fraction, isNull);
  });

  test('an HTTP error is reported and leaves no files behind', () async {
    status = HttpStatus.notFound;
    await expectLater(
      downloader.download(PretrainedModel.compact, directory: workDir.path),
      throwsA(
        isA<ModelDownloadException>().having(
          (e) => e.statusCode,
          'statusCode',
          HttpStatus.notFound,
        ),
      ),
    );

    expect(workDir.listSync(), isEmpty);
  });

  test('a response that is not a model is rejected', () async {
    // What a captive portal or an error page would send.
    body = utf8.encode('<html><body>Not found</body></html>');
    await expectLater(
      downloader.download(PretrainedModel.compact, directory: workDir.path),
      throwsA(isA<ModelDownloadException>()),
    );

    expect(
      workDir.listSync(),
      isEmpty,
      reason: 'the partial file must be cleaned up',
    );
  });

  test('a truncated file is rejected', () async {
    body = buildSyntheticModel().sublist(0, 4);
    await expectLater(
      downloader.download(PretrainedModel.compact, directory: workDir.path),
      throwsA(isA<ModelDownloadException>()),
    );
  });

  test('a body cut short without a Content-Length is rejected', () async {
    // The dangerous shape: the stream ends normally, and the eight header
    // bytes are already there, so nothing downstream notices. The file used
    // to be renamed into place and then returned from the cache forever.
    sendContentLength = false;
    cutAfter = 20;

    await expectLater(
      downloader.download(PretrainedModel.compact, directory: workDir.path),
      throwsA(isA<ModelDownloadException>()),
    );
    expect(workDir.listSync(), isEmpty);
  });

  test('a body cut short of its Content-Length is rejected', () async {
    cutAfter = 20;

    await expectLater(
      downloader.download(PretrainedModel.compact, directory: workDir.path),
      throwsA(isA<ModelDownloadException>()),
    );
    expect(workDir.listSync(), isEmpty);
  });

  test(
    'a model already on disk that is not a model is fetched again',
    () async {
      final target = downloader.fileIn(workDir.path, PretrainedModel.compact);
      await target.writeAsString('<html>error</html>');

      final file = await downloader.download(
        PretrainedModel.compact,
        directory: workDir.path,
      );

      expect(requestCount, 1, reason: 'the broken file must not be kept');
      expect(await file.readAsBytes(), body);
    },
  );

  test('a server that goes quiet does not hang the caller', () async {
    goQuiet = true;
    final impatient = ModelDownloader(
      baseUrl: Uri.parse('http://${server.address.host}:${server.port}/'),
      connectionTimeout: const Duration(milliseconds: 200),
      stallTimeout: const Duration(milliseconds: 200),
    );

    try {
      await expectLater(
        impatient.download(PretrainedModel.compact, directory: workDir.path),
        throwsA(isA<ModelDownloadException>()),
      );
    } finally {
      await impatient.close();
    }
  });

  test('a compressed response does not confuse progress', () async {
    // The announced length is the one on the wire; what arrives is what it
    // unpacks to. Measuring one against the other put progress at 600% and
    // would now reject a download that is perfectly fine.
    compress = true;
    final seen = <DownloadProgress>[];
    final file = await downloader.download(
      PretrainedModel.compact,
      directory: workDir.path,
      onProgress: seen.add,
    );

    expect(await file.readAsBytes(), body);
    expect(seen.last.totalBytes, isNull);
    expect(seen.last.fraction, isNull);
  });

  test('two downloads at once do not spoil each other', () async {
    // They used to share one scratch file, interleave their chunks into it,
    // and rename the mixture into place.
    final second = ModelDownloader(
      baseUrl: Uri.parse('http://${server.address.host}:${server.port}/'),
    );

    try {
      final files = await Future.wait([
        downloader.download(PretrainedModel.compact, directory: workDir.path),
        second.download(PretrainedModel.compact, directory: workDir.path),
      ]);

      for (final file in files) {
        expect(await file.readAsBytes(), body);
      }
      expect(
        workDir.listSync().whereType<File>().map((f) => f.path).toList(),
        [endsWith('lid.176.ftz')],
        reason: 'no scratch files left behind',
      );
    } finally {
      await second.close();
    }
  });

  test(
    'a mistake in onProgress is not reported as a download failure',
    () async {
      // Catching everything used to rewrite the caller's own bug into a
      // ModelDownloadException, without the stack that says where it was.
      await expectLater(
        downloader.download(
          PretrainedModel.compact,
          directory: workDir.path,
          onProgress: (_) => throw StateError('a bug in my callback'),
        ),
        throwsStateError,
      );
      expect(workDir.listSync(), isEmpty);
    },
  );

  test('fileIn and urlOf point where download will act', () {
    expect(
      downloader.fileIn(workDir.path, PretrainedModel.full).path,
      endsWith('lid.176.bin'),
    );
    expect(
      downloader.urlOf(PretrainedModel.full).toString(),
      endsWith('/lid.176.bin'),
    );
  });

  test('fileIn takes a directory the way the filesystem does', () {
    // Resolving the name as a URI normalized `..` by text, which is only
    // right when nothing on the way is a symlink, and left an empty
    // directory meaning something other than "here".
    expect(downloader.fileIn('', PretrainedModel.compact).path, 'lid.176.ftz');
    expect(
      downloader.fileIn('a/../b', PretrainedModel.compact).path,
      contains('a/../b'),
    );
    expect(
      downloader.fileIn('${workDir.path}/', PretrainedModel.compact).path,
      '${workDir.path}/lid.176.ftz',
    );
  });

  test('a closed downloader refuses to work', () async {
    await downloader.close();
    expect(
      () => downloader.download(
        PretrainedModel.compact,
        directory: workDir.path,
      ),
      throwsStateError,
    );
  });

  test('loadOrDownloadModel fetches and then reads from disk', () async {
    final identifier = await loadOrDownloadModel(
      PretrainedModel.compact,
      directory: workDir.path,
      downloader: downloader,
    );

    expect(identifier.languages, ['x', 'y']);
    expect(identifier.identify('alpha')!.label, 'x');
    expect(requestCount, 1);

    await loadOrDownloadModel(
      PretrainedModel.compact,
      directory: workDir.path,
      downloader: downloader,
    );
    expect(requestCount, 1);
  });
}
