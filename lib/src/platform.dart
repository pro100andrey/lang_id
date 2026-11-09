/// `true` on the web, where `int` is a double: no 64-bit integer
/// arithmetic, and SIMD types emulated in software.
///
/// A constant, so each build keeps only its own branch wherever this is
/// tested. Defined once because the branches it picks must agree: the FNV
/// hash multiplies one way or the other depending on it.
const bool numbersAreDoubles = identical(1, 1.0);
