package co.immanuel.arkspeech.core.nlu

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TextNormalizerTest {
    @Test fun `native digits become ascii`() {
        assertEquals("2500 रुपये", TextNormalizer.canonical("२५०० रुपये"))
        assertEquals("10 মণি", TextNormalizer.canonical("১০ মণি"))
        assertEquals("98765", TextNormalizer.canonical("௯௮௭௬௫"))
    }

    @Test fun `indic words with combining marks stay single tokens`() {
        val t = NormalizedText("डॉक्टर शर्मा के साथ, नाळை")
        assertEquals(listOf("डॉक्टर", "शर्मा", "के", "साथ", ",", "नाळை"), t.tokens.map { it.text })
    }

    @Test fun `phrase search respects word boundaries`() {
        val t = NormalizedText("Please book an appointment at the bookshop")
        assertTrue(t.containsPhrase("book"))
        assertTrue(t.containsPhrase(TextNormalizer.phrase("Book an appointment")))
        assertFalse(t.containsPhrase("shop"))
        assertFalse(NormalizedText("the bookshop").containsPhrase("book"))
    }

    @Test fun `phrase normalisation strips punctuation and case`() {
        assertEquals("phone number is", TextNormalizer.phrase("  Phone number is: "))
        assertEquals("के पास", TextNormalizer.phrase("के पास"))
    }

    @Test fun `case preserved in text and lowered for matching`() {
        val t = NormalizedText("Doctor Sharma")
        assertEquals("Doctor Sharma", t.text)
        assertEquals("doctor sharma", t.lower)
    }
}
