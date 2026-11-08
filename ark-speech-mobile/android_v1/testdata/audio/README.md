# Test audio

16 kHz mono 16-bit WAV clips used by the core unit tests (signal validation) and the Android
instrumented tests (end-to-end pipeline). The speech clips were synthesised with macOS `say`
voices (Rishi/Tara/Aman en-IN, Lekha hi-IN, Vani ta-IN, Geeta te-IN, Soumya kn-IN, Piya bn-IN);
see `scripts/gen-test-audio.sh`. Synthetic voices are a smoke test, not a benchmark of real speech.

| file | content |
|---|---|
| en_appointment.wav | "Book an appointment with Doctor Sharma for tomorrow at ten in the morning. My phone number is 98765 43210." |
| en_payment.wav | "Record a payment of two thousand five hundred rupees to Ramesh by UPI on the fifth of November." |
| en_issue.wav | "I want to report a water supply problem near Gandhi Road bus stop. It is urgent." |
| hi_appointment.wav | मुझे कल सुबह दस बजे डॉक्टर शर्मा के साथ अपॉइंटमेंट बुक करनी है। मेरा फ़ोन नंबर … है। |
| hi_payment.wav | रमेश को दो हज़ार पाँच सौ रुपये यूपीआई से भेजे हैं, पाँच नवंबर को। |
| hi_issue.wav | गांधी रोड बस स्टॉप के पास पानी की सप्लाई की समस्या है, बहुत ज़रूरी है। |
| hi_codemix.wav | मुझे Doctor Sharma के साथ kal appointment book करनी है, time ten AM. |
| hi_appointment_noisy.wav | hi_appointment with white noise mixed in |
| ta_appointment.wav | நாளை காலை பத்து மணிக்கு டாக்டர் குமாருடன் அப்பாய்ண்ட்மெண்ட் புக் செய்ய வேண்டும். |
| ta_issue.wav | காந்தி சாலை பேருந்து நிறுத்தம் அருகில் தண்ணீர் வசதி பிரச்சனை உள்ளது. இது அவசரம். |
| te_appointment.wav | రేపు ఉదయం పది గంటలకు డాక్టర్ రావుతో అపాయింట్‌మెంట్ బుక్ చేయాలి. |
| kn_stock.wav | ಗೋದಾಮಿಗೆ ಐವತ್ತು ಕಿಲೋ ಅಕ್ಕಿ ಸೇರಿಸಿ. |
| bn_payment.wav | রমেশকে পাঁচই নভেম্বর ইউপিআই-এর মাধ্যমে দুই হাজার পাঁচশো টাকা পাঠানো হয়েছে। |
| edge_silence_4s.wav | digital silence |
| edge_noise_4s.wav | white noise |
| edge_too_short.wav | 0.3 s of speech |
| edge_clipped.wav | speech amplified 12x (heavy clipping) |
| edge_very_quiet.wav | speech at -34 dB |
| edge_speech_then_silence.wav | speech followed by 3 s silence |
| edge_stereo_44k.wav | 44.1 kHz stereo WAV (must be converted) |
| edge_unsupported.m4a | AAC in M4A (must be rejected as unsupported) |
