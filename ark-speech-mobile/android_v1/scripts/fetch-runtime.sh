#!/usr/bin/env bash
# Downloads the pinned sherpa-onnx Android AAR into app/libs (it is not committed: 37 MB).
# The "static-link-onnxruntime" variant links ONNX Runtime into libsherpa-onnx-jni.so so the app can
# also use the official onnxruntime-android AAR (for Anuvani-LID) without a duplicate libonnxruntime.so.
set -euo pipefail
VERSION="${SHERPA_ONNX_VERSION:-2.0.1}"
SHA256_EXPECTED="07e042d97f3a6bd5b7fe74d4eef0513dc01a714b16c15a903250095aba43a575"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/app/libs/sherpa-onnx-static-link-onnxruntime-$VERSION.aar"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/v$VERSION/sherpa-onnx-static-link-onnxruntime-$VERSION.aar"
mkdir -p "$ROOT/app/libs"
if [ -f "$DEST" ]; then echo "already present: $DEST"; else
  echo "downloading $URL"; curl -L --fail --retry 3 -o "$DEST.part" "$URL"; mv "$DEST.part" "$DEST"
fi
if [ "$VERSION" = "2.0.1" ]; then
  ACTUAL="$(shasum -a 256 "$DEST" | awk '{print $1}')"
  if [ "$ACTUAL" != "$SHA256_EXPECTED" ]; then echo "SHA-256 mismatch: $ACTUAL" >&2; exit 1; fi
  echo "sha256 verified"
fi
echo "ok: $DEST"
