#!/usr/bin/env bash
# Downloads model packs from app/src/main/assets/models/catalog.json into ./models/<pack-id>/ and,
# with --push, copies them to a connected device/emulator where the app looks for them:
#   /sdcard/Android/data/co.immanuel.arkspeech/files/models/<pack-id>/
# Usage: scripts/fetch-models.sh [--push] [--list] [pack-id ...]
# Default packs: silero-vad anuvani-lid-int8 omnilingual-300m-ctc-int8 (the "default" tier, ~510 MB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CATALOG="$ROOT/app/src/main/assets/models/catalog.json"
OUT="$ROOT/models"
PKG="co.immanuel.arkspeech"
PUSH=0; LIST=0; PACKS=()
for a in "$@"; do case "$a" in --push) PUSH=1;; --list) LIST=1;; *) PACKS+=("$a");; esac; done
if [ "$LIST" = 1 ]; then
  python3 -I -c "import json,sys; c=json.load(open(sys.argv[1])); [print(f\"{p['id']:30s} {p['kind']:4s} {p['installedSizeBytes']/1048576:8.1f} MB  {p['tier']:8s} {p['licence']:11s} {p['displayName']}\") for p in c['packs']]" "$CATALOG"; exit 0
fi
[ ${#PACKS[@]} -eq 0 ] && PACKS=(silero-vad anuvani-lid-int8 omnilingual-300m-ctc-int8)
mkdir -p "$OUT"
for id in "${PACKS[@]}"; do
  python3 -I - "$CATALOG" "$id" "$OUT" <<'PY'
import json,sys,os,hashlib,urllib.request
cat,pid,out=sys.argv[1:4]
packs={p["id"]:p for p in json.load(open(cat))["packs"]}
p=packs.get(pid) or sys.exit(f"unknown pack {pid}; use --list")
d=os.path.join(out,pid); os.makedirs(d,exist_ok=True)
for f in p["files"]:
    dest=os.path.join(d,f["name"])
    if os.path.isfile(dest) and (not f.get("sizeBytes") or os.path.getsize(dest)==f["sizeBytes"]):
        print(f"  present  {pid}/{f['name']}"); continue
    print(f"  fetching {pid}/{f['name']} ({f.get('sizeBytes',0)/1048576:.1f} MB)")
    req=urllib.request.Request(f["url"],headers={"User-Agent":"ark-speech-fetch/1"})
    with urllib.request.urlopen(req,timeout=120) as r, open(dest+".part","wb") as o:
        h=hashlib.sha256(); n=0
        while True:
            b=r.read(1<<20)
            if not b: break
            o.write(b); h.update(b); n+=len(b)
            print(f"\r    {n/1048576:8.1f} MB",end="",flush=True)
    print()
    if f.get("sha256") and h.hexdigest()!=f["sha256"]:
        os.remove(dest+".part"); sys.exit(f"SHA-256 mismatch for {f['name']}")
    os.replace(dest+".part",dest)
print(f"ok: {pid}")
PY
done
if [ "$PUSH" = 1 ]; then
  adb get-state >/dev/null 2>&1 || { echo "no device connected (adb)"; exit 1; }
  # The app must have run once so the app-specific external directory exists and is writable by adb.
  for id in "${PACKS[@]}"; do
    DEST="/sdcard/Android/data/$PKG/files/models/$id"
    echo "pushing $id -> $DEST"
    adb shell mkdir -p "$DEST"
    adb push "$OUT/$id/." "$DEST/" >/dev/null
  done
  echo "pushed. Open the Models tab (or restart the app) to pick them up."
fi
