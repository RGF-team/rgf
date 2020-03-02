# Changelog

## 0.7 (Feb 2020)

  - Added support for compilation with Clang and AppleClang.

## 0.6 (Feb 2018)

  - Fixed bug which led to program crash in case of usage of small samples weights.

## 0.5 (Sept 2017)

  - Added OpenMP support and multithreading for discretization.
  - Added loop unrolling and compilation option for simd optimization.

## 0.4 (Aug 2017)

  - Fixed bug which truncated negative float values to `numeric_limits<float>::min()`, causing degration in prediction performance for datasets with negative values; changed truncation to `numeric_limits<float>::lowest()`.

## 0.3 (Dec 2016)

  - Fixed several bugs that affect prediction performance (especially for small datasets).

## 0.2 (Aug 2016)

  - This is the first release.

    It only supports binary classification and regression, with significant simplifications from the [original RGF algorithm](https://arxiv.org/abs/1109.0887) for speed consideration.

    Additional functionalities  will be supported in future releases.
