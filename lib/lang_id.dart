/// Language identification in pure Dart.
///
/// Reads fastText models — both plain `.bin` and quantized `.ftz` — and runs
/// inference itself, with no FFI and no native code. Works on the VM, in AOT
/// builds and on the web.
///
/// ```dart
/// import 'dart:io';
/// import 'package:lang_id/lang_id.dart';
///
/// final model = LanguageIdentifier.fromBytes(
///     File('models/lid.176.ftz').readAsBytesSync());
/// print(model.identify('Hello, how are you?')?.label); // en
/// ```
///
/// Model weights are not part of this package and are distributed under
/// CC BY-SA 3.0 — see `NOTICE.md`.
library;

export 'src/language_identifier.dart' show LanguageIdentifier, ModelInfo;
export 'src/model_args.dart' show FastTextArchitecture, FastTextLoss, ModelArgs;
export 'src/prediction.dart' show Prediction;
