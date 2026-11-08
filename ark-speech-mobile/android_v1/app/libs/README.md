# Native runtime

`sherpa-onnx-static-link-onnxruntime-<version>.aar` is downloaded here by `scripts/fetch-runtime.sh`
(it is not committed). The "static-link" variant links ONNX Runtime into `libsherpa-onnx-jni.so`, so
the app can also depend on the official `onnxruntime-android` AAR (used for Anuvani-LID) without two
copies of `libonnxruntime.so` colliding. SHA-256 of the pinned file is in the script.
