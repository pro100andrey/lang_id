import 'dart:typed_data';

/// Rounds an intermediate result down to float32.
///
/// fastText runs inference in `float` while Dart computes in `double`. On
/// confident predictions the difference is invisible, but in the tail of the
/// output, where probabilities bottom out at `1e-5`, the extra precision
/// swaps neighbouring labels around. To match the original bit for bit, every
/// compound expression is rounded exactly where C++ rounds it.
///
/// Single operations on two float32 values need no rounding: an addition or a
/// multiplication computed in double and stored into a [Float32List] is
/// already correctly rounded.
double float32(double value) {
  _scratch[0] = value;

  return _scratch[0];
}

final _scratch = Float32List(1);
