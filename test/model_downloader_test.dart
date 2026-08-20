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

  setUp(() async {
    body = buildSyntheticModel();
    status = HttpStatus.ok;
    requestCount = 0;
    sendContentLength = true;

    server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    unawaited(() async {
      await for (final request in server) {
        requestCount++;
        request.response.statusCode = status;
        if (sendContentLength) {
          request.response.contentLength = body.length;
        }
        request.response.add(body);
        await request.response.close();
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
