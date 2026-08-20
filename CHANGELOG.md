## 0.1.0

- First release.
- Reads fastText models: `.bin` (dense matrix) and `.ftz` (product quantization).
- Inference for supervised models with any loss: hierarchical softmax, plain
  softmax, negative sampling, one-vs-all.
- `sentenceVector` exposes the embedding behind a prediction.
- Bit-exact parity with the reference fastText implementation, checked for both
  predictions and sentence vectors on `lid.176.ftz` and `lid.176.bin`.
- `ModelDownloader` and `loadOrDownloadModel` fetch the weights into a
  directory of your choosing: the package carries no model of its own.
- No FFI and no native code: runs on the VM, in AOT builds and on the web.
