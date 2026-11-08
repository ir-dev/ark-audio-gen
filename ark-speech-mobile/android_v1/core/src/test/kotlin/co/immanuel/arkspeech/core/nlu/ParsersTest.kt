package co.immanuel.arkspeech.core.nlu

import co.immanuel.arkspeech.core.config.TemporalDirection
import co.immanuel.arkspeech.core.support.TestSupport
import java.time.LocalDate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ParsersTest {
    private val today: LocalDate = TestSupport.clock.today() // 2026-11-08, a Sunday

    private fun numbers(s: String) = NumberParser.findAll(NormalizedText(s)).map { it.value }

    @Test fun `english number words compose`() {
        assertEquals(listOf(2500.0), numbers("two thousand five hundred"))
        assertEquals(listOf(120000.0), numbers("one lakh twenty thousand"))
        assertEquals(listOf(50.0), numbers("fifty kilos"))
        assertEquals(listOf(2500.0), numbers("2,500 rupees"))
        assertEquals(listOf(2.5), numbers("2.5 litres"))
    }

    @Test fun `hindi number words compose in devanagari and roman script`() {
        assertEquals(listOf(2500.0), numbers("दो हज़ार पाँच सौ"))
        assertEquals(listOf(2500.0), numbers("dhai hazaar"))
        assertEquals(listOf(120000.0), numbers("ek lakh bees hazaar"))
        assertEquals(listOf(99.0), numbers("निन्यानवे"))
        assertEquals(listOf(1500.0), numbers("डेढ़ हज़ार"))
    }

    @Test fun `tamil telugu kannada bengali number words`() {
        assertEquals(listOf(2500.0), numbers("இரண்டாயிரம் ஐநூறு"))
        assertEquals(listOf(50.0), numbers("ಐವತ್ತು"))
        assertEquals(listOf(2500.0), numbers("দুই হাজার পাঁচশো"))
        assertEquals(listOf(10.0), numbers("పది"))
    }

    @Test fun `spoken digit sequences are kept as digits`() {
        val m = NumberParser.findAll(NormalizedText("call nine eight seven six five four three two one zero now")).single()
        assertTrue(m.isDigitSequence)
        assertEquals("9876543210", m.digits)
    }

    @Test fun `relative dates resolve against the fixed clock`() {
        fun dates(s: String, dir: TemporalDirection = TemporalDirection.FUTURE) = DateParser.findAll(NormalizedText(s), today, dir).map { it.value }.distinct().sorted()
        assertEquals(listOf("2026-11-09"), dates("tomorrow"))
        assertEquals(listOf("2026-11-10"), dates("day after tomorrow"))
        assertEquals(listOf("2026-11-09"), dates("कल सुबह"))
        assertEquals(listOf("2026-11-07", "2026-11-09"), dates("कल", TemporalDirection.UNSPECIFIED))
        assertEquals(listOf("2026-11-09"), dates("நாளை காலை"))
        assertEquals(listOf("2026-11-09"), dates("రేపు"))
        assertEquals(listOf("2026-11-09"), dates("ನಾಳೆ"))
        assertEquals(listOf("2026-11-07"), dates("kal", TemporalDirection.PAST))
    }

    @Test fun `explicit dates in several scripts`() {
        fun dates(s: String, dir: TemporalDirection = TemporalDirection.PAST) = DateParser.findAll(NormalizedText(s), today, dir).map { it.value }.distinct()
        assertEquals(listOf("2026-11-05"), dates("on the fifth of November"))
        assertEquals(listOf("2026-11-05"), dates("पाँच नवंबर को"))
        assertEquals(listOf("2026-11-05"), dates("পাঁচই নভেম্বর"))
        assertEquals(listOf("2026-11-12"), dates("12/11/2026"))
        assertEquals(listOf("2026-11-12"), dates("2026-11-12"))
        assertEquals(listOf("2027-03-15"), dates("15 March", TemporalDirection.FUTURE))
        assertEquals(listOf("2026-03-15"), dates("15 March", TemporalDirection.PAST))
    }

    @Test fun `weekday resolves to next occurrence for future fields`() {
        val c = DateParser.findAll(NormalizedText("next Monday"), today, TemporalDirection.FUTURE).single()
        assertEquals("2026-11-09", c.value)
        val p = DateParser.findAll(NormalizedText("सोमवार"), today, TemporalDirection.PAST).single()
        assertEquals("2026-11-02", p.value)
    }

    @Test fun `times with periods and markers`() {
        fun time(s: String) = TimeParser.findAll(NormalizedText(s)).maxByOrNull { it.confidence }
        assertEquals("10:00", time("at ten in the morning")!!.value)
        assertEquals("10:00", time("सुबह दस बजे")!!.value)
        assertEquals("17:00", time("शाम पाँच बजे")!!.value)
        assertEquals("22:30", time("10:30 pm")!!.value)
        assertEquals("15:00", time("3 pm")!!.value)
        assertEquals("10:00", time("காலை பத்து மணிக்கு")!!.value)
        assertEquals("10:00", time("ఉదయం పది గంటలకు")!!.value)
        assertEquals("10:30", time("साढ़े दस बजे")!!.value)
        assertEquals("09:45", time("पौने दस बजे")!!.value)
        assertEquals("10:00", time("time ten AM")!!.value)
        assertEquals("14:30", time("14:30")!!.value)
    }

    @Test fun `plain hour without period is low confidence`() {
        val c = TimeParser.findAll(NormalizedText("come at 10 baje")).single()
        assertEquals("10:00", c.value)
        assertTrue(c.confidence < 0.6f)
        assertNotNull(c.note)
    }

    @Test fun `phone numbers from digits and words`() {
        fun phone(s: String) = PhoneParser.findAll(NormalizedText(s)).maxByOrNull { it.confidence }?.value
        assertEquals("9876543210", phone("my number is nine eight seven six five four three two one zero"))
        assertEquals("9876543210", phone("call 98765 43210"))
        assertEquals("9876543210", phone("+91 98765-43210"))
        assertEquals("9876543210", phone("नौ आठ सात छह पाँच चार तीन दो एक शून्य"))
        assertEquals("9987654321", phone("double nine eight seven six five four three two one"))
        assertNull(phone("call me at ten"))
    }

    @Test fun `amounts with currency words`() {
        fun amount(s: String) = AmountParser.findAll(NormalizedText(s), allowBare = false).maxByOrNull { it.confidence }
        assertEquals("2500", amount("two thousand five hundred rupees")!!.value)
        assertEquals("2500", amount("paid ₹2,500 today")!!.value)
        assertEquals("2500", amount("दो हज़ार पाँच सौ रुपये")!!.value)
        assertEquals("2500", amount("দুই হাজার পাঁচশো টাকা")!!.value)
        assertEquals("INR", amount("Rs 500")!!.extra["currency"])
        assertNull(amount("fifty kilos of rice"))
    }

    @Test fun `durations to ISO 8601`() {
        fun dur(s: String) = DurationParser.findAll(NormalizedText(s)).firstOrNull()?.value
        assertEquals("PT2H", dur("for two hours"))
        assertEquals("PT30M", dur("30 minutes"))
        assertEquals("P3D", dur("तीन दिन"))
        assertEquals("PT1H30M", dur("डेढ़ घंटे"))
    }
}
