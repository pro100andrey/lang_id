/// fastText classifiers in pure Dart, language identification among them.
///
/// Reads supervised fastText models — both plain `.bin` and quantized `.ftz`
/// — and runs inference itself, with no FFI and no native code. Works on the
/// VM, in AOT builds and on the web.
///
/// ```dart
/// import 'dart:io';
/// import 'package:lang_id/lang_id.dart';
///
/// final model = FastTextClassifier.fromBytes(
///     File('models/lid.176.ftz').readAsBytesSync());
/// print(model.classify('Hello, how are you?')?.label); // en
/// ```
///
/// Model weights are not part of this package and are distributed under
/// CC BY-SA 3.0 — see `NOTICE.md`.
library;

export 'src/fasttext_classifier.dart' show FastTextClassifier, ModelInfo;
export 'src/model_args.dart' show FastTextArchitecture, FastTextLoss, ModelArgs;
export 'src/prediction.dart' show Prediction;
