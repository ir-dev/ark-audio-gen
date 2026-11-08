#!/usr/bin/env bash
# Runs the on-device benchmark (BenchmarkTest) on the connected device, then pulls the JSON/markdown
# report and captures memory and battery statistics from the system.
# Prerequisites: models pushed (scripts/fetch-models.sh --push), device unlocked, ideally unplugged
# after the test APKs are installed (battery deltas are meaningless while charging).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="co.immanuel.arkspeech"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/benchmarks/results/$STAMP"; mkdir -p "$OUT"
adb get-state >/dev/null 2>&1 || { echo "no device connected"; exit 1; }
adb shell getprop ro.product.model > "$OUT/device.txt"; adb shell getprop ro.build.version.release >> "$OUT/device.txt"
adb shell dumpsys batterystats --reset >/dev/null 2>&1 || true
cd "$ROOT"
./gradlew --console=plain :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=$PKG.BenchmarkTest | tee "$OUT/gradle.log"
adb pull "/sdcard/Android/data/$PKG/files/benchmark/benchmark.json" "$OUT/" || echo "no benchmark.json (test skipped? models installed?)"
adb pull "/sdcard/Android/data/$PKG/files/benchmark/benchmark.md" "$OUT/" || true
adb shell dumpsys meminfo "$PKG" > "$OUT/meminfo.txt" 2>/dev/null || true
adb shell dumpsys batterystats "$PKG" > "$OUT/batterystats.txt" 2>/dev/null || true
adb shell du -sh "/sdcard/Android/data/$PKG/files/models/"* > "$OUT/model-storage.txt" 2>/dev/null || true
adb logcat -d -s Benchmark:I > "$OUT/logcat-benchmark.txt" || true
echo "results in $OUT"; [ -f "$OUT/benchmark.md" ] && cat "$OUT/benchmark.md"
