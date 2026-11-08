package co.immanuel.arkspeech.core.nlu

/**
 * Hand-built lexicons for the typed parsers: number words, relative days, weekdays, months, time
 * periods, currency and duration units. Coverage is deepest for English, Hindi (Devanagari and
 * romanised "Hinglish") and Tamil, with the core vocabulary for Telugu, Kannada, Bengali, Marathi and
 * Gujarati. Administrators extend coverage per field through synonyms and anchors, not by editing code.
 */
object Lexicons {

    enum class NumKind { UNIT, HUNDRED, MULTIPLIER, FRACTION }

    class NumWord(val value: Double, val kind: NumKind)

    private fun units(vararg pairs: Pair<String, Int>) = pairs.associate { it.first to NumWord(it.second.toDouble(), NumKind.UNIT) }

    private val en = units(
        "zero" to 0, "oh" to 0, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6, "seven" to 7, "eight" to 8,
        "nine" to 9, "ten" to 10, "eleven" to 11, "twelve" to 12, "thirteen" to 13, "fourteen" to 14, "fifteen" to 15, "sixteen" to 16,
        "seventeen" to 17, "eighteen" to 18, "nineteen" to 19, "twenty" to 20, "thirty" to 30, "forty" to 40, "fifty" to 50, "sixty" to 60,
        "seventy" to 70, "eighty" to 80, "ninety" to 90,
    ) + mapOf(
        "hundred" to NumWord(100.0, NumKind.HUNDRED), "thousand" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "lakh" to NumWord(100_000.0, NumKind.MULTIPLIER), "lakhs" to NumWord(100_000.0, NumKind.MULTIPLIER), "lac" to NumWord(100_000.0, NumKind.MULTIPLIER),
        "crore" to NumWord(10_000_000.0, NumKind.MULTIPLIER), "crores" to NumWord(10_000_000.0, NumKind.MULTIPLIER),
        "million" to NumWord(1_000_000.0, NumKind.MULTIPLIER), "half" to NumWord(0.5, NumKind.FRACTION), "quarter" to NumWord(0.25, NumKind.FRACTION),
    )

    private val hiUnits = listOf(
        "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
        "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
        "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
        "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
        "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
        "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
        "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
        "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
        "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे",
        "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
    ).mapIndexed { i, w -> w to NumWord(i.toDouble(), NumKind.UNIT) }.toMap() + mapOf(
        "पांच" to NumWord(5.0, NumKind.UNIT), "छः" to NumWord(6.0, NumKind.UNIT), "छे" to NumWord(6.0, NumKind.UNIT), "पन्द्रह" to NumWord(15.0, NumKind.UNIT),
        "ज़ीरो" to NumWord(0.0, NumKind.UNIT), "जीरो" to NumWord(0.0, NumKind.UNIT),
        "सौ" to NumWord(100.0, NumKind.HUNDRED), "हज़ार" to NumWord(1_000.0, NumKind.MULTIPLIER), "हजार" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "लाख" to NumWord(100_000.0, NumKind.MULTIPLIER), "करोड़" to NumWord(10_000_000.0, NumKind.MULTIPLIER), "करोड" to NumWord(10_000_000.0, NumKind.MULTIPLIER),
        "आधा" to NumWord(0.5, NumKind.FRACTION), "डेढ़" to NumWord(1.5, NumKind.UNIT), "ढाई" to NumWord(2.5, NumKind.UNIT), "सवा" to NumWord(1.25, NumKind.UNIT),
    )

    /** Romanised Hindi as people type and ASR sometimes emits it. */
    private val hiLatin = units(
        "ek" to 1, "do" to 2, "teen" to 3, "tin" to 3, "char" to 4, "chaar" to 4, "paanch" to 5, "panch" to 5, "chhe" to 6, "che" to 6, "cheh" to 6,
        "saat" to 7, "sat" to 7, "aath" to 8, "ath" to 8, "nau" to 9, "das" to 10, "gyarah" to 11, "barah" to 12, "terah" to 13, "chaudah" to 14,
        "pandrah" to 15, "solah" to 16, "satrah" to 17, "atharah" to 18, "unnis" to 19, "bees" to 20, "pachees" to 25, "tees" to 30, "chalis" to 40,
        "pachas" to 50, "saath" to 60, "sattar" to 70, "assi" to 80, "nabbe" to 90,
    ) + mapOf(
        "sau" to NumWord(100.0, NumKind.HUNDRED), "hazaar" to NumWord(1_000.0, NumKind.MULTIPLIER), "hazar" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "lakh" to NumWord(100_000.0, NumKind.MULTIPLIER), "karod" to NumWord(10_000_000.0, NumKind.MULTIPLIER), "crore" to NumWord(10_000_000.0, NumKind.MULTIPLIER),
        "dhai" to NumWord(2.5, NumKind.UNIT), "dhaai" to NumWord(2.5, NumKind.UNIT), "dedh" to NumWord(1.5, NumKind.UNIT), "sava" to NumWord(1.25, NumKind.UNIT),
        "aadha" to NumWord(0.5, NumKind.FRACTION),
    )

    private val ta = units(
        "பூஜ்ஜியம்" to 0, "ஒன்று" to 1, "ஒரு" to 1, "இரண்டு" to 2, "மூன்று" to 3, "நான்கு" to 4, "ஐந்து" to 5, "ஆறு" to 6, "ஏழு" to 7, "எட்டு" to 8,
        "ஒன்பது" to 9, "பத்து" to 10, "பதினொன்று" to 11, "பன்னிரண்டு" to 12, "பதின்மூன்று" to 13, "பதினான்கு" to 14, "பதினைந்து" to 15,
        "பதினாறு" to 16, "பதினேழு" to 17, "பதினெட்டு" to 18, "பத்தொன்பது" to 19, "இருபது" to 20, "முப்பது" to 30, "நாற்பது" to 40,
        "ஐம்பது" to 50, "அறுபது" to 60, "எழுபது" to 70, "எண்பது" to 80, "தொண்ணூறு" to 90,
        // common fused hundreds and thousands
        "இருநூறு" to 200, "முந்நூறு" to 300, "நானூறு" to 400, "ஐநூறு" to 500, "அறுநூறு" to 600, "எழுநூறு" to 700, "எண்ணூறு" to 800, "தொள்ளாயிரம்" to 900,
        "இரண்டாயிரம்" to 2000, "மூவாயிரம்" to 3000, "நாலாயிரம்" to 4000, "ஐயாயிரம்" to 5000, "பத்தாயிரம்" to 10000,
    ) + mapOf(
        "நூறு" to NumWord(100.0, NumKind.HUNDRED), "ஆயிரம்" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "லட்சம்" to NumWord(100_000.0, NumKind.MULTIPLIER), "கோடி" to NumWord(10_000_000.0, NumKind.MULTIPLIER), "அரை" to NumWord(0.5, NumKind.FRACTION),
    )

    private val te = units(
        "సున్నా" to 0, "ఒకటి" to 1, "ఒక" to 1, "రెండు" to 2, "మూడు" to 3, "నాలుగు" to 4, "ఐదు" to 5, "ఆరు" to 6, "ఏడు" to 7, "ఎనిమిది" to 8,
        "తొమ్మిది" to 9, "పది" to 10, "పదకొండు" to 11, "పన్నెండు" to 12, "ఇరవై" to 20, "ముప్పై" to 30, "నలభై" to 40, "యాభై" to 50, "అరవై" to 60,
        "డెబ్బై" to 70, "ఎనభై" to 80, "తొంభై" to 90,
    ) + mapOf("వంద" to NumWord(100.0, NumKind.HUNDRED), "వెయ్యి" to NumWord(1_000.0, NumKind.MULTIPLIER), "వేలు" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "లక్ష" to NumWord(100_000.0, NumKind.MULTIPLIER), "కోటి" to NumWord(10_000_000.0, NumKind.MULTIPLIER))

    private val kn = units(
        "ಸೊನ್ನೆ" to 0, "ಒಂದು" to 1, "ಎರಡು" to 2, "ಮೂರು" to 3, "ನಾಲ್ಕು" to 4, "ಐದು" to 5, "ಆರು" to 6, "ಏಳು" to 7, "ಎಂಟು" to 8, "ಒಂಬತ್ತು" to 9,
        "ಹತ್ತು" to 10, "ಹನ್ನೊಂದು" to 11, "ಹನ್ನೆರಡು" to 12, "ಇಪ್ಪತ್ತು" to 20, "ಮೂವತ್ತು" to 30, "ನಲವತ್ತು" to 40, "ಐವತ್ತು" to 50, "ಅರವತ್ತು" to 60,
        "ಎಪ್ಪತ್ತು" to 70, "ಎಂಬತ್ತು" to 80, "ತೊಂಬತ್ತು" to 90,
    ) + mapOf("ನೂರು" to NumWord(100.0, NumKind.HUNDRED), "ಸಾವಿರ" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "ಲಕ್ಷ" to NumWord(100_000.0, NumKind.MULTIPLIER), "ಕೋಟಿ" to NumWord(10_000_000.0, NumKind.MULTIPLIER))

    private val bn = units(
        "শূন্য" to 0, "এক" to 1, "দুই" to 2, "তিন" to 3, "চার" to 4, "পাঁচ" to 5, "ছয়" to 6, "সাত" to 7, "আট" to 8, "নয়" to 9, "দশ" to 10,
        "এগারো" to 11, "বারো" to 12, "পনেরো" to 15, "বিশ" to 20, "কুড়ি" to 20, "পঁচিশ" to 25, "ত্রিশ" to 30, "চল্লিশ" to 40, "পঞ্চাশ" to 50,
        "ষাট" to 60, "সত্তর" to 70, "আশি" to 80, "নব্বই" to 90,
        "একশো" to 100, "দুশো" to 200, "তিনশো" to 300, "চারশো" to 400, "পাঁচশো" to 500, "ছয়শো" to 600, "সাতশো" to 700, "আটশো" to 800, "নয়শো" to 900,
    ) + mapOf("শত" to NumWord(100.0, NumKind.HUNDRED), "শো" to NumWord(100.0, NumKind.HUNDRED), "হাজার" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "লাখ" to NumWord(100_000.0, NumKind.MULTIPLIER), "লক্ষ" to NumWord(100_000.0, NumKind.MULTIPLIER), "কোটি" to NumWord(10_000_000.0, NumKind.MULTIPLIER))

    private val mr = units(
        "शून्य" to 0, "एक" to 1, "दोन" to 2, "तीन" to 3, "चार" to 4, "पाच" to 5, "सहा" to 6, "सात" to 7, "आठ" to 8, "नऊ" to 9, "दहा" to 10,
        "वीस" to 20, "पंचवीस" to 25, "तीस" to 30, "चाळीस" to 40, "पन्नास" to 50, "साठ" to 60, "सत्तर" to 70, "ऐंशी" to 80, "नव्वद" to 90,
    ) + mapOf("शंभर" to NumWord(100.0, NumKind.HUNDRED), "हजार" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "लाख" to NumWord(100_000.0, NumKind.MULTIPLIER), "कोटी" to NumWord(10_000_000.0, NumKind.MULTIPLIER))

    private val gu = units(
        "શૂન્ય" to 0, "એક" to 1, "બે" to 2, "ત્રણ" to 3, "ચાર" to 4, "પાંચ" to 5, "છ" to 6, "સાત" to 7, "આઠ" to 8, "નવ" to 9, "દસ" to 10,
        "વીસ" to 20, "પચીસ" to 25, "ત્રીસ" to 30, "ચાલીસ" to 40, "પચાસ" to 50, "સાઠ" to 60, "સિત્તેર" to 70, "એંસી" to 80, "નેવું" to 90,
    ) + mapOf("સો" to NumWord(100.0, NumKind.HUNDRED), "હજાર" to NumWord(1_000.0, NumKind.MULTIPLIER),
        "લાખ" to NumWord(100_000.0, NumKind.MULTIPLIER), "કરોડ" to NumWord(10_000_000.0, NumKind.MULTIPLIER))

    /** One merged map: words are script-distinct across languages, and romanised Hindi overlaps English only on "lakh"/"crore". */
    val numberWords: Map<String, NumWord> = en + hiLatin + hiUnits + ta + te + kn + bn + mr + gu

    /** Single digits by word, for spoken phone numbers and codes. */
    val digitWords: Map<String, Int> = numberWords.filter { it.value.kind == NumKind.UNIT && it.value.value in 0.0..9.0 && it.value.value % 1.0 == 0.0 }
        .mapValues { it.value.value.toInt() }

    val repeatWords: Map<String, Int> = mapOf("double" to 2, "triple" to 3, "डबल" to 2, "ट्रिपल" to 3)

    /** Relative-day words → day offsets; more than one offset means the word is ambiguous (Hindi "kal"). */
    val relativeDays: Map<String, List<Int>> = mapOf(
        "today" to listOf(0), "tonight" to listOf(0), "tomorrow" to listOf(1), "yesterday" to listOf(-1),
        "day after tomorrow" to listOf(2), "day after" to listOf(2), "day before yesterday" to listOf(-2),
        "आज" to listOf(0), "कल" to listOf(1, -1), "परसों" to listOf(2, -2), "नरसों" to listOf(3, -3),
        "aaj" to listOf(0), "kal" to listOf(1, -1), "parson" to listOf(2, -2), "parso" to listOf(2, -2),
        "இன்று" to listOf(0), "இன்றைக்கு" to listOf(0), "நாளை" to listOf(1), "நாளைக்கு" to listOf(1), "நேற்று" to listOf(-1), "நாளை மறுநாள்" to listOf(2),
        "ఈరోజు" to listOf(0), "ఈ రోజు" to listOf(0), "రేపు" to listOf(1), "నిన్న" to listOf(-1), "ఎల్లుండి" to listOf(2),
        "ಇಂದು" to listOf(0), "ನಾಳೆ" to listOf(1), "ನಿನ್ನೆ" to listOf(-1), "ನಾಡಿದ್ದು" to listOf(2),
        "আজ" to listOf(0), "আজকে" to listOf(0), "আগামীকাল" to listOf(1), "কাল" to listOf(1, -1), "গতকাল" to listOf(-1), "পরশু" to listOf(2, -2),
        "उद्या" to listOf(1), "काल" to listOf(-1), "परवा" to listOf(2, -2),
        "આજે" to listOf(0), "કાલે" to listOf(1, -1), "આવતીકાલે" to listOf(1), "ગઈકાલે" to listOf(-1), "પરમ દિવસે" to listOf(2),
    )

    /** Weekday names → java.time DayOfWeek value (1 = Monday). */
    val weekdays: Map<String, Int> = mapOf(
        "monday" to 1, "tuesday" to 2, "wednesday" to 3, "thursday" to 4, "friday" to 5, "saturday" to 6, "sunday" to 7,
        "सोमवार" to 1, "मंगलवार" to 2, "बुधवार" to 3, "गुरुवार" to 4, "बृहस्पतिवार" to 4, "शुक्रवार" to 5, "शनिवार" to 6, "रविवार" to 7, "इतवार" to 7,
        "somvar" to 1, "mangalvar" to 2, "budhvar" to 3, "guruvar" to 4, "shukravar" to 5, "shanivar" to 6, "ravivar" to 7,
        "திங்கள்" to 1, "செவ்வாய்" to 2, "புதன்" to 3, "வியாழன்" to 4, "வெள்ளி" to 5, "சனி" to 6, "ஞாயிறு" to 7,
        "సోమవారం" to 1, "మంగళవారం" to 2, "బుధవారం" to 3, "గురువారం" to 4, "శుక్రవారం" to 5, "శనివారం" to 6, "ఆదివారం" to 7,
        "ಸೋಮವಾರ" to 1, "ಮಂಗಳವಾರ" to 2, "ಬುಧವಾರ" to 3, "ಗುರುವಾರ" to 4, "ಶುಕ್ರವಾರ" to 5, "ಶನಿವಾರ" to 6, "ಭಾನುವಾರ" to 7,
        "সোমবার" to 1, "মঙ্গলবার" to 2, "বুধবার" to 3, "বৃহস্পতিবার" to 4, "শুক্রবার" to 5, "শনিবার" to 6, "রবিবার" to 7,
    )

    val months: Map<String, Int> = mapOf(
        "january" to 1, "jan" to 1, "february" to 2, "feb" to 2, "march" to 3, "mar" to 3, "april" to 4, "apr" to 4, "may" to 5, "june" to 6, "jun" to 6,
        "july" to 7, "jul" to 7, "august" to 8, "aug" to 8, "september" to 9, "sept" to 9, "sep" to 9, "october" to 10, "oct" to 10,
        "november" to 11, "nov" to 11, "december" to 12, "dec" to 12,
        "जनवरी" to 1, "फ़रवरी" to 2, "फरवरी" to 2, "मार्च" to 3, "अप्रैल" to 4, "मई" to 5, "जून" to 6, "जुलाई" to 7, "अगस्त" to 8,
        "सितंबर" to 9, "सितम्बर" to 9, "अक्टूबर" to 10, "अक्तूबर" to 10, "नवंबर" to 11, "नवम्बर" to 11, "दिसंबर" to 12, "दिसम्बर" to 12,
        "ஜனவரி" to 1, "பிப்ரவரி" to 2, "மார்ச்" to 3, "ஏப்ரல்" to 4, "மே" to 5, "ஜூன்" to 6, "ஜூலை" to 7, "ஆகஸ்ட்" to 8, "செப்டம்பர்" to 9,
        "அக்டோபர்" to 10, "நவம்பர்" to 11, "டிசம்பர்" to 12,
        "జనవరి" to 1, "ఫిబ్రవరి" to 2, "మార్చి" to 3, "ఏప్రిల్" to 4, "మే" to 5, "జూన్" to 6, "జూలై" to 7, "ఆగస్టు" to 8, "సెప్టెంబర్" to 9,
        "అక్టోబర్" to 10, "నవంబర్" to 11, "డిసెంబర్" to 12,
        "ಜನವರಿ" to 1, "ಫೆಬ್ರವರಿ" to 2, "ಮಾರ್ಚ್" to 3, "ಏಪ್ರಿಲ್" to 4, "ಮೇ" to 5, "ಜೂನ್" to 6, "ಜುಲೈ" to 7, "ಆಗಸ್ಟ್" to 8, "ಸೆಪ್ಟೆಂಬರ್" to 9,
        "ಅಕ್ಟೋಬರ್" to 10, "ನವೆಂಬರ್" to 11, "ಡಿಸೆಂಬರ್" to 12,
        "জানুয়ারি" to 1, "ফেব্রুয়ারি" to 2, "মার্চ" to 3, "এপ্রিল" to 4, "মে" to 5, "জুন" to 6, "জুলাই" to 7, "আগস্ট" to 8, "সেপ্টেম্বর" to 9,
        "অক্টোবর" to 10, "নভেম্বর" to 11, "ডিসেম্বর" to 12,
    )

    /** Ordinal day words used in spoken dates ("the fifth of November", "पाँचवीं तारीख"). */
    val ordinalWords: Map<String, Int> = mapOf(
        "first" to 1, "second" to 2, "third" to 3, "fourth" to 4, "fifth" to 5, "sixth" to 6, "seventh" to 7, "eighth" to 8, "ninth" to 9, "tenth" to 10,
        "eleventh" to 11, "twelfth" to 12, "thirteenth" to 13, "fourteenth" to 14, "fifteenth" to 15, "sixteenth" to 16, "seventeenth" to 17, "eighteenth" to 18,
        "nineteenth" to 19, "twentieth" to 20, "twenty first" to 21, "twenty second" to 22, "twenty third" to 23, "twenty fourth" to 24, "twenty fifth" to 25,
        "twenty sixth" to 26, "twenty seventh" to 27, "twenty eighth" to 28, "twenty ninth" to 29, "thirtieth" to 30, "thirty first" to 31,
        "पहली" to 1, "पहला" to 1, "दूसरी" to 2, "दूसरा" to 2, "तीसरी" to 3, "तीसरा" to 3, "चौथी" to 4, "चौथा" to 4, "पाँचवीं" to 5, "पांचवीं" to 5, "पाँचवां" to 5,
        "छठी" to 6, "छठा" to 6, "सातवीं" to 7, "सातवां" to 7, "आठवीं" to 8, "आठवां" to 8, "नौवीं" to 9, "नौवां" to 9, "दसवीं" to 10, "दसवां" to 10,
        "pehli" to 1, "doosri" to 2, "teesri" to 3, "chauthi" to 4, "paanchvi" to 5,
    )

    /** Bengali spoken dates carry an ordinal suffix: "পাঁচই" = the 5th. */
    val bengaliOrdinalDays: Map<String, Int> = mapOf(
        "পহেলা" to 1, "দোসরা" to 2, "তেসরা" to 3, "চৌঠা" to 4, "পাঁচই" to 5, "ছয়ই" to 6, "সাতই" to 7, "আটই" to 8, "নয়ই" to 9, "দশই" to 10,
        "এগারোই" to 11, "বারোই" to 12, "তেরোই" to 13, "চৌদ্দই" to 14, "পনেরোই" to 15, "ষোলোই" to 16, "সতেরোই" to 17, "আঠারোই" to 18, "উনিশে" to 19,
        "বিশে" to 20, "একুশে" to 21, "বাইশে" to 22, "তেইশে" to 23, "চব্বিশে" to 24, "পঁচিশে" to 25, "ছাব্বিশে" to 26, "সাতাশে" to 27, "আঠাশে" to 28,
        "ঊনত্রিশে" to 29, "ত্রিশে" to 30, "একত্রিশে" to 31,
    )

    enum class DayPeriod { MORNING, AFTERNOON, EVENING, NIGHT }

    val dayPeriods: Map<String, DayPeriod> = mapOf(
        "morning" to DayPeriod.MORNING, "am" to DayPeriod.MORNING, "a m" to DayPeriod.MORNING, "noon" to DayPeriod.AFTERNOON, "afternoon" to DayPeriod.AFTERNOON,
        "evening" to DayPeriod.EVENING, "pm" to DayPeriod.EVENING, "p m" to DayPeriod.EVENING, "night" to DayPeriod.NIGHT, "midnight" to DayPeriod.NIGHT,
        "सुबह" to DayPeriod.MORNING, "सवेरे" to DayPeriod.MORNING, "दोपहर" to DayPeriod.AFTERNOON, "शाम" to DayPeriod.EVENING, "रात" to DayPeriod.NIGHT,
        "subah" to DayPeriod.MORNING, "dopahar" to DayPeriod.AFTERNOON, "shaam" to DayPeriod.EVENING, "sham" to DayPeriod.EVENING, "raat" to DayPeriod.NIGHT,
        "காலை" to DayPeriod.MORNING, "மதியம்" to DayPeriod.AFTERNOON, "மாலை" to DayPeriod.EVENING, "இரவு" to DayPeriod.NIGHT,
        "ఉదయం" to DayPeriod.MORNING, "మధ్యాహ్నం" to DayPeriod.AFTERNOON, "సాయంత్రం" to DayPeriod.EVENING, "రాత్రి" to DayPeriod.NIGHT,
        "ಬೆಳಿಗ್ಗೆ" to DayPeriod.MORNING, "ಮಧ್ಯಾಹ್ನ" to DayPeriod.AFTERNOON, "ಸಂಜೆ" to DayPeriod.EVENING, "ರಾತ್ರಿ" to DayPeriod.NIGHT,
        "সকাল" to DayPeriod.MORNING, "সকালে" to DayPeriod.MORNING, "দুপুর" to DayPeriod.AFTERNOON, "দুপুরে" to DayPeriod.AFTERNOON, "সন্ধ্যা" to DayPeriod.EVENING, "সন্ধ্যায়" to DayPeriod.EVENING, "রাত" to DayPeriod.NIGHT, "রাতে" to DayPeriod.NIGHT,
        "सकाळी" to DayPeriod.MORNING, "दुपारी" to DayPeriod.AFTERNOON, "संध्याकाळी" to DayPeriod.EVENING, "रात्री" to DayPeriod.NIGHT,
        "સવારે" to DayPeriod.MORNING, "બપોરે" to DayPeriod.AFTERNOON, "સાંજે" to DayPeriod.EVENING, "રાત્રે" to DayPeriod.NIGHT,
    )

    /** Words that mark an hour: "10 baje", "பத்து மணிக்கு", "o'clock". */
    val hourMarkers: Set<String> = setOf(
        "o'clock", "oclock", "बजे", "बजकर", "baje", "மணி", "மணிக்கு", "గంటలకు", "గంటకు", "ಗಂಟೆಗೆ", "টায়", "টা", "वाजता", "વાગ્યે",
    )

    /** Hindi/Marathi fractional hour prefixes: साढ़े दस = 10:30, सवा दस = 10:15, पौने दस = 9:45. */
    val hourFractionPrefixes: Map<String, Int> = mapOf("साढ़े" to 30, "साढे" to 30, "sadhe" to 30, "सवा" to 15, "sava" to 15, "पौने" to -15, "paune" to -15)

    val currencyWords: Set<String> = setOf(
        "rupees", "rupee", "rs", "inr", "₹", "रुपये", "रुपए", "रुपया", "रुपैया", "रु", "rupaye", "rupay", "rupiya", "rupaiya",
        "ரூபாய்", "ரூபா", "ரூ", "రూపాయలు", "రూపాయి", "ರೂಪಾಯಿ", "ರೂ", "টাকা", "રૂપિયા", "روپے", "روپیہ", "ടാക്കാ", "രൂപ",
    )

    enum class DurationUnit(val isoUnit: Char, val isTime: Boolean) {
        MINUTE('M', true), HOUR('H', true), DAY('D', false), WEEK('W', false), MONTH('M', false)
    }

    val durationUnits: Map<String, DurationUnit> = mapOf(
        "minute" to DurationUnit.MINUTE, "minutes" to DurationUnit.MINUTE, "min" to DurationUnit.MINUTE, "mins" to DurationUnit.MINUTE,
        "hour" to DurationUnit.HOUR, "hours" to DurationUnit.HOUR, "hr" to DurationUnit.HOUR, "hrs" to DurationUnit.HOUR,
        "day" to DurationUnit.DAY, "days" to DurationUnit.DAY, "week" to DurationUnit.WEEK, "weeks" to DurationUnit.WEEK, "month" to DurationUnit.MONTH, "months" to DurationUnit.MONTH,
        "मिनट" to DurationUnit.MINUTE, "घंटे" to DurationUnit.HOUR, "घंटा" to DurationUnit.HOUR, "घण्टे" to DurationUnit.HOUR, "दिन" to DurationUnit.DAY, "हफ्ते" to DurationUnit.WEEK, "हफ़्ते" to DurationUnit.WEEK, "सप्ताह" to DurationUnit.WEEK, "महीने" to DurationUnit.MONTH, "महीना" to DurationUnit.MONTH,
        "ghante" to DurationUnit.HOUR, "ghanta" to DurationUnit.HOUR, "din" to DurationUnit.DAY, "hafte" to DurationUnit.WEEK, "mahine" to DurationUnit.MONTH,
        "நிமிடம்" to DurationUnit.MINUTE, "நிமிடங்கள்" to DurationUnit.MINUTE, "மணிநேரம்" to DurationUnit.HOUR, "நாள்" to DurationUnit.DAY, "நாட்கள்" to DurationUnit.DAY, "வாரம்" to DurationUnit.WEEK, "மாதம்" to DurationUnit.MONTH,
        "నిమిషాలు" to DurationUnit.MINUTE, "గంటలు" to DurationUnit.HOUR, "రోజులు" to DurationUnit.DAY, "వారాలు" to DurationUnit.WEEK, "నెలలు" to DurationUnit.MONTH,
        "ನಿಮಿಷ" to DurationUnit.MINUTE, "ಗಂಟೆ" to DurationUnit.HOUR, "ದಿನ" to DurationUnit.DAY, "ವಾರ" to DurationUnit.WEEK, "ತಿಂಗಳು" to DurationUnit.MONTH,
        "মিনিট" to DurationUnit.MINUTE, "ঘণ্টা" to DurationUnit.HOUR, "দিন" to DurationUnit.DAY, "সপ্তাহ" to DurationUnit.WEEK, "মাস" to DurationUnit.MONTH,
    )

    val yesWords: Set<String> = setOf("yes", "yeah", "yep", "haan", "han", "ji", "हाँ", "हां", "जी", "ஆம்", "ஆமாம்", "అవును", "ಹೌದು", "হ্যাঁ", "হাঁ", "होय", "હા")
    val noWords: Set<String> = setOf("no", "nope", "nahi", "nahin", "नहीं", "नही", "இல்லை", "కాదు", "లేదు", "ಇಲ್ಲ", "না", "नाही", "ના")
}
