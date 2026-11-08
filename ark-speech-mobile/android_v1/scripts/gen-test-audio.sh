#!/usr/bin/env bash
# Regenerates testdata/audio with macOS system voices (say) + Python for the synthetic edge cases.
# macOS only; the committed WAVs were produced with this script on macOS 26 (voices: Rishi/Tara/Aman en-IN,
# Lekha hi-IN, Vani ta-IN, Geeta te-IN, Soumya kn-IN, Piya bn-IN).
set -euo pipefail
OUT="$(cd "$(dirname "$0")/.." && pwd)/testdata/audio"; mkdir -p "$OUT"
gen() { say -v "$2" --file-format=WAVE --data-format=LEI16@16000 -o "$OUT/$1.wav" "$3"; }
gen en_appointment Rishi "Book an appointment with Doctor Sharma for tomorrow at ten in the morning. My phone number is nine eight seven six five four three two one zero."
gen en_payment Tara "Record a payment of two thousand five hundred rupees to Ramesh by UPI on the fifth of November."
gen en_issue Aman "I want to report a water supply problem near Gandhi Road bus stop. It is urgent."
gen hi_appointment Lekha "मुझे कल सुबह दस बजे डॉक्टर शर्मा के साथ अपॉइंटमेंट बुक करनी है। मेरा फ़ोन नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है।"
gen hi_payment Lekha "रमेश को दो हज़ार पाँच सौ रुपये यूपीआई से भेजे हैं, पाँच नवंबर को।"
gen hi_issue Lekha "गांधी रोड बस स्टॉप के पास पानी की सप्लाई की समस्या है, बहुत ज़रूरी है।"
gen hi_codemix Lekha "मुझे Doctor Sharma के साथ kal appointment book करनी है, time ten AM."
gen ta_appointment Vani "நாளை காலை பத்து மணிக்கு டாக்டர் குமாருடன் அப்பாய்ண்ட்மெண்ட் புக் செய்ய வேண்டும்."
gen ta_issue Vani "காந்தி சாலை பேருந்து நிறுத்தம் அருகில் தண்ணீர் வசதி பிரச்சனை உள்ளது. இது அவசரம்."
gen te_appointment Geeta "రేపు ఉదయం పది గంటలకు డాక్టర్ రావుతో అపాయింట్‌మెంట్ బుక్ చేయాలి."
gen kn_stock Soumya "ಗೋದಾಮಿಗೆ ಐವತ್ತು ಕಿಲೋ ಅಕ್ಕಿ ಸೇರಿಸಿ."
gen bn_payment Piya "রমেশকে পাঁচই নভেম্বর ইউপিআই-এর মাধ্যমে দুই হাজার পাঁচশো টাকা পাঠানো হয়েছে।"
afconvert -f WAVE -d LEI16@44100 -c 2 "$OUT/en_appointment.wav" "$OUT/edge_stereo_44k.wav"
afconvert -f m4af -d aac "$OUT/en_appointment.wav" "$OUT/edge_unsupported.m4a"
python3 -I - "$OUT" <<'PY'
import wave,struct,random,sys
F=sys.argv[1]
def read(p):
    with wave.open(p,'rb') as w: return list(struct.unpack('<%dh'%w.getnframes(), w.readframes(w.getnframes())))
def write(p,s):
    with wave.open(p,'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(struct.pack('<%dh'%len(s), *[max(-32768,min(32767,int(x))) for x in s]))
random.seed(7)
write(f"{F}/edge_silence_4s.wav",[0]*64000)
write(f"{F}/edge_noise_4s.wav",[random.gauss(0,0.3)*32767 for _ in range(64000)])
en=read(f"{F}/en_appointment.wav"); hi=read(f"{F}/hi_appointment.wav")
write(f"{F}/edge_too_short.wav", en[:4800]); write(f"{F}/edge_clipped.wav",[x*12 for x in en]); write(f"{F}/edge_very_quiet.wav",[x*0.02 for x in en])
write(f"{F}/hi_appointment_noisy.wav",[x+random.gauss(0,0.06)*32767 for x in hi]); write(f"{F}/edge_speech_then_silence.wav", en+[0]*48000)
PY
echo "written to $OUT"
