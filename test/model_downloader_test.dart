@TestOn('vm')
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

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
    final classifier = await loadOrDownloadModel(
      PretrainedModel.compact,
      directory: workDir.path,
      downloader: downloader,
    );

    expect(classifier.labels, ['x', 'y']);
    expect(classifier.classify('alpha')!.label, 'x');
    expect(requestCount, 1);

    await loadOrDownloadModel(
      PretrainedModel.compact,
      directory: workDir.path,
      downloader: downloader,
    );
    expect(requestCount, 1);
  });

  group('when a download goes wrong', () {
    /// A server that speaks raw TCP, for the cases [HttpServer] is too
    /// well-behaved to produce. [onConnect] gets each accepted socket.
    Future<ServerSocket> rawServer(
      void Function(Socket socket) onConnect,
    ) async {
      final raw = await ServerSocket.bind(InternetAddress.loopbackIPv4, 0);
      raw.listen(onConnect);
      addTearDown(raw.close);

      return raw;
    }

    ModelDownloader impatientAt(int port) => ModelDownloader(
      baseUrl: Uri.parse('http://127.0.0.1:$port/'),
      connectionTimeout: const Duration(milliseconds: 300),
      stallTimeout: const Duration(milliseconds: 300),
    );

    test(
      'a server that never answers has its connection dropped on timeout',
      () async {
        // The caller used to be told about the timeout while the socket,
        // and with it the process, stayed open for as long as the server
        // cared to hold it.
        final closed = Completer<void>();
        final raw = await rawServer((socket) {
          socket.listen((_) {}, onDone: closed.complete, onError: (_) {});
        });
        final impatient = impatientAt(raw.port);
        try {
          await expectLater(
            impatient.download(
              PretrainedModel.compact,
              directory: workDir.path,
            ),
            throwsA(isA<ModelDownloadException>()),
          );
          await closed.future.timeout(const Duration(seconds: 5));
        } finally {
          await impatient.close();
        }
      },
    );

    test(
      'an error page that stalls is still reported, with its status',
      () async {
        final raw = await rawServer((socket) {
          socket
            ..listen((_) {}, onError: (_) {})
            ..write(
              'HTTP/1.1 503 Service Unavailable\r\n'
              'Content-Length: 1000\r\n\r\nsorry',
            );
        });
        final impatient = impatientAt(raw.port);
        try {
          await expectLater(
            impatient
                .download(PretrainedModel.compact, directory: workDir.path)
                .timeout(const Duration(seconds: 5)),
            throwsA(
              isA<ModelDownloadException>().having(
                (e) => e.statusCode,
                'statusCode',
                HttpStatus.serviceUnavailable,
              ),
            ),
          );
        } finally {
          await impatient.close();
        }
      },
    );

    test(
      'a directory that cannot be written is a download exception',
      () async {
        // A failed write used to surface as an unhandled error that killed
        // the isolate. The scratch file is now opened before the request, so
        // the failure also comes before anything is asked of the server:
        // an unread response would hold its socket, and the process, open.
        final locked = Directory(pathIn('locked'))..createSync();
        await Process.run('chmod', ['555', locked.path]);
        addTearDown(() => Process.run('chmod', ['755', locked.path]));

        await expectLater(
          downloader.download(PretrainedModel.compact, directory: locked.path),
          throwsA(isA<ModelDownloadException>()),
        );
        expect(requestCount, 0);
      },
      skip: Platform.isWindows
          ? 'needs POSIX permissions'
          : _runsAsRoot()
          ? 'root writes through the permission bits'
          : false,
    );

    test(
      'a directory that cannot be created is a download exception',
      () async {
        // A file where the directory should be: no permission bits involved,
        // so this holds for root as well. It used to escape as a raw
        // FileSystemException.
        final blocker = File(pathIn('blocker'))..writeAsStringSync('');

        for (final directory in [blocker.path, '${blocker.path}/below']) {
          await expectLater(
            downloader.download(PretrainedModel.compact, directory: directory),
            throwsA(isA<ModelDownloadException>()),
          );
        }
        expect(requestCount, 0);
      },
    );

    test('an empty directory means the working one', () async {
      // In a process of its own: the working directory belongs to the whole
      // process, and changing it here would pull it out from under every
      // suite that finds its models by a relative path.
      final script = File(pathIn('empty_directory.dart'))
        ..writeAsStringSync('''
import 'package:lang_id/lang_id_io.dart';

Future<void> main(List<String> args) async {
  final downloader = ModelDownloader(baseUrl: Uri.parse(args.single));
  try {
    final file = await downloader.download(
      PretrainedModel.compact,
      directory: '',
    );
    print(file.path);
  } finally {
    await downloader.close();
  }
}
''');
      final packages = File('.dart_tool/package_config.json').absolute.path;
      final run = await Process.run(Platform.resolvedExecutable, [
        '--packages=$packages',
        script.path,
        'http://${server.address.host}:${server.port}/',
      ], workingDirectory: workDir.path);

      expect(run.exitCode, 0, reason: '${run.stdout}${run.stderr}');
      expect((run.stdout as String).trim(), 'lid.176.ftz');
      expect(File(pathIn('lid.176.ftz')).readAsBytesSync(), body);
    });

    test(
      'downloads from two isolates get scratch files of their own',
      () async {
        // The name used to be the process id and a counter, and the counter is
        // per isolate: two isolates of one process both chose `.<pid>.0.part`.
        // The server holds both requests, so both scratch files are open at
        // once.
        final held = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
        addTearDown(() => held.close(force: true));
        final waiting = <HttpRequest>[];
        final bothArrived = Completer<void>();
        held.listen((request) {
          waiting.add(request);
          if (waiting.length == 2) {
            bothArrived.complete();
          }
        });

        final base = 'http://${held.address.host}:${held.port}/';
        final directory = workDir.path;
        Future<void> fetch() => _downloadInIsolate(base, directory);
        final downloads = [fetch(), fetch()];

        await bothArrived.future.timeout(const Duration(seconds: 10));
        final scratch = workDir
            .listSync()
            .where((entry) => entry.path.endsWith('.part'))
            .toList();
        expect(scratch, hasLength(2));

        for (final request in waiting) {
          request.response
            ..contentLength = body.length
            ..add(body);
          await request.response.close();
        }
        await Future.wait(downloads);
        expect(File(pathIn('lid.176.ftz')).readAsBytesSync(), body);
      },
    );

    test(
      'stale scratch files are found however the directory is spelled',
      () async {
        final stale = File(pathIn('lid.176.ftz.1.0.part'))
          ..writeAsStringSync('left over')
          ..setLastModifiedSync(
            DateTime.now().subtract(const Duration(hours: 2)),
          );

        await downloader.download(
          PretrainedModel.compact,
          directory: '${workDir.path}//',
        );
        expect(stale.existsSync(), isFalse);
      },
    );

    test('loadOrDownloadModel heals a cached model cut short', () async {
      // 0.1.0 could cache a body cut in half behind a good header. The
      // header check passes it, and only the full read notices.
      File(
        pathIn('lid.176.ftz'),
      ).writeAsBytesSync(body.sublist(0, body.length - 5));

      final classifier = await loadOrDownloadModel(
        PretrainedModel.compact,
        directory: workDir.path,
        downloader: downloader,
      );
      expect(classifier.classify('alpha')!.label, 'x');
      expect(requestCount, 1);
    });

    test('the wait for a response is a stall, not a connection', () async {
      // A short connect timeout on the client used to cut off a server
      // that was merely slow to start answering.
      final slow = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => slow.close(force: true));
      slow.listen((request) async {
        await Future<void>.delayed(const Duration(milliseconds: 600));
        request.response
          ..contentLength = body.length
          ..add(body);
        await request.response.close();
      });
      final patient = ModelDownloader(
        baseUrl: Uri.parse('http://${slow.address.host}:${slow.port}/'),
        connectionTimeout: const Duration(milliseconds: 200),
      );
      addTearDown(patient.close);

      final file = await patient.download(
        PretrainedModel.compact,
        directory: workDir.path,
      );
      expect(file.readAsBytesSync(), body);
    });

    test(
      'loadOrDownloadModel reports a fresh download that is broken',
      () async {
        // An error page in the cache is thrown away and fetched again by
        // download itself. If what comes back is broken too, that is the
        // answer: it used to count as the cache and be fetched a third time.
        File(pathIn('lid.176.ftz')).writeAsStringSync('<html>not a model');
        body = body.sublist(0, body.length - 5);

        await expectLater(
          loadOrDownloadModel(
            PretrainedModel.compact,
            directory: workDir.path,
            downloader: downloader,
          ),
          throwsFormatException,
        );
        expect(requestCount, 1);
      },
    );

    test('a client of your own keeps its connection timeout', () {
      final client = HttpClient()
        ..connectionTimeout = const Duration(minutes: 2);
      addTearDown(client.close);

      expect(
        ModelDownloader(httpClient: client).connectionTimeout,
        const Duration(minutes: 2),
      );
      expect(
        ModelDownloader(
          httpClient: client,
          connectionTimeout: const Duration(seconds: 5),
        ).connectionTimeout,
        const Duration(seconds: 5),
      );
      expect(
        ModelDownloader(httpClient: HttpClient()).connectionTimeout,
        const Duration(seconds: 30),
      );
    });

    test('a base URL without a trailing slash keeps its last segment', () {
      expect(
        ModelDownloader(
          baseUrl: Uri.parse('https://mirror.example/fasttext'),
        ).urlOf(PretrainedModel.compact).toString(),
        'https://mirror.example/fasttext/lid.176.ftz',
      );
    });
  });
}

/// Whether the suite runs as root, which writes through permission bits.
bool _runsAsRoot() {
  try {
    final id = Process.runSync('id', ['-u']);

    return (id.stdout as String).trim() == '0';
  } on ProcessException {
    return false;
  }
}

/// Downloads the compact model from [base] in an isolate of its own. A
/// top-level function, so that the closure sent along captures nothing but
/// its two strings.
Future<void> _downloadInIsolate(String base, String directory) => Isolate.run(
  () async {
    final downloader = ModelDownloader(baseUrl: Uri.parse(base));
    try {
      await downloader.download(PretrainedModel.compact, directory: directory);
    } finally {
      await downloader.close();
    }
  },
);
