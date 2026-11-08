#!/usr/bin/env bash
# Runs the instrumented tests on the connected device WITHOUT Gradle's post-run uninstall, which would
# delete the app's external files (and the model packs with them). Builds nothing: run
# `./gradlew :app:assembleDebug :app:assembleDebugAndroidTest` first.
# Usage: [ASR_MODEL=<pack-id>] scripts/instrumented-tests.sh [TestClass ...]   (default: all classes; ASR_MODEL routes all languages to one pack)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; PKG="co.immanuel.arkspeech"
APK="$ROOT/app/build/outputs/apk/debug/app-debug.apk"; TAPK="$ROOT/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
[ -f "$APK" ] && [ -f "$TAPK" ] || { echo "build the APKs first: ./gradlew :app:assembleDebug :app:assembleDebugAndroidTest"; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "no device connected"; exit 1; }
for a in "$APK" "$TAPK"; do
  out="$(adb install -r -t "$a" 2>&1)" || { echo "install failed for $a: $out"; exit 1; }
  echo "$out" | grep -q Success || { echo "install did not report Success for $a: $out"; exit 1; }
done
echo "installed app + test APK ($(date +%H:%M:%S))"
adb shell pm grant $PKG android.permission.RECORD_AUDIO 2>/dev/null || true
# Make sure the app's directories exist and the default model packs are present.
adb shell am start -W -n $PKG/.ui.MainActivity >/dev/null 2>&1 || true; sleep 2
if ! adb shell "ls /sdcard/Android/data/$PKG/files/models/silero-vad/silero_vad.onnx" >/dev/null 2>&1; then
  echo "models missing on device; pushing default packs"; "$ROOT/scripts/fetch-models.sh" --push
fi
CLASSES="${*:-co.immanuel.arkspeech.SherpaVadTest,co.immanuel.arkspeech.AnuvaniLidTest,co.immanuel.arkspeech.PipelineE2ETest,co.immanuel.arkspeech.BenchmarkTest}"
CLASSES="${CLASSES// /,}"
OUT="$ROOT/benchmarks/results/instrumented-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
adb logcat -c || true
EXTRA=""; [ -n "${ASR_MODEL:-}" ] && EXTRA="-e asrModel $ASR_MODEL" && echo "routing every language to $ASR_MODEL for this run"
adb shell am instrument -w -r -e class "$CLASSES" $EXTRA $PKG.test/androidx.test.runner.AndroidJUnitRunner | tee "$OUT/instrument.txt" \
  | grep -E '^INSTRUMENTATION_STATUS: (test|class)=|^INSTRUMENTATION_STATUS_CODE|^INSTRUMENTATION_RESULT|^INSTRUMENTATION_CODE|Time:|OK \(|FAILURES|Tests run' || true
adb logcat -d -s PipelineE2ETest:I AnuvaniLidTest:I Benchmark:I TestRunner:E > "$OUT/logcat.txt" || true
adb pull "/sdcard/Android/data/$PKG/files/e2e" "$OUT/" >/dev/null 2>&1 || true
adb pull "/sdcard/Android/data/$PKG/files/benchmark" "$OUT/" >/dev/null 2>&1 || true
echo "outputs in $OUT"
