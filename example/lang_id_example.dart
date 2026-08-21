// Identifies the language of a text.
//
//   dart run example/lang_id_example.dart                      # built-in demo
//   dart run example/lang_id_example.dart "Привіт, як справи?"  # your own text
//   cat file.txt | dart run example/lang_id_example.dart -      # line by line
//
// Standard input is read only when asked for with `-`, so running the file
// with no arguments always finishes instead of waiting for input that may
// never come.
//
// No setup step: the model is fetched on the first run into ./models and read
// from disk afterwards.
import 'dart:convert';
import 'dart:io';

import 'package:lang_id/lang_id_io.dart';

/// Shown when the example is run with nothing to classify.
const _demoTexts = [
  'Привіт, як справи? Сьогодні чудова погода.',
  'The quick brown fox jumps over the lazy dog',
  'Bonjour tout le monde, comment allez-vous ?',
  'Die Katze sitzt auf der Matte und schläft',
  'El rápido zorro marrón salta sobre el perro perezoso',
  '今日はいい天気ですね',
  '中文是世界上使用人数最多的语言',
  'ภาษาไทยเป็นภาษาราชการของประเทศไทย',
  'اللغة العربية من أقدم اللغات السامية',
  'Kiswahili ni lugha inayozungumzwa Afrika Mashariki',
];

Future<void> main(List<String> args) async {
  final LanguageIdentifier identifier;
  try {
    identifier = await loadOrDownloadModel(
      PretrainedModel.compact,
      directory: 'models',
      onProgress: (progress) => stderr.write('\r$progress'),
    );
  } on ModelDownloadException catch (error) {
    // The first run needs the network. Saying so is more use than a stack
    // trace and an exit code of 255.
    stderr.writeln('\nCould not fetch the model: $error');
    exitCode = 1;

    return;
  } on FormatException catch (error) {
    stderr
      ..writeln('\nmodels/lid.176.ftz is not readable: ${error.message}')
      ..writeln('Delete it and run again to fetch a fresh copy.');
    exitCode = 1;

    return;
  }

  stderr.writeln('\n${identifier.info}\n');

  if (args.isEmpty) {
    _demo(identifier);
    return;
  }

  if (args.length == 1 && (args.single == '-' || args.single == '--stdin')) {
    await _readStdin(identifier);
    return;
  }

  _report(identifier, args.join(' '));
}

Future<void> _readStdin(LanguageIdentifier identifier) async {
  // Malformed bytes become replacement characters rather than an error.
  // Text of an unknown encoding is exactly what a language identifier is
  // reached for, and the strict decoder would abandon the whole stream over
  // one Latin-1 byte, reporting nothing for the lines that were fine.
  final lines = stdin
      .transform(const Utf8Decoder(allowMalformed: true))
      .transform(const LineSplitter());
  await for (final line in lines) {
    if (line.trim().isNotEmpty) {
      _report(identifier, line);
    }
  }
}

void _demo(LanguageIdentifier identifier) {
  stdout.writeln(
    '${identifier.languages.length} languages known, '
    'top 3 for each sample:\n',
  );

  for (final text in _demoTexts) {
    _report(identifier, text);
  }

  final vector = identifier.sentenceVector(_demoTexts.first);
  final head = vector.take(6).map((v) => v.toStringAsFixed(3)).join(' ');
  stdout
    ..writeln(
      '\nThe embedding behind that first prediction '
      '(${vector.length} values):',
    )
    ..writeln('$head  …')
    ..writeln(
      '\nPass your own text as an argument, '
      'or pipe lines in with a trailing "-".',
    );
}

void _report(LanguageIdentifier identifier, String text) {
  final top = identifier.predict(text, k: 3, threshold: 0.01);
  final verdict = top.isEmpty ? '?' : top.join(', ');
  stdout.writeln('${verdict.padRight(30)}$text');
}
