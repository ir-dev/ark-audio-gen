package co.immanuel.arkspeech.core.util

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/** Injected time source so date parsing ("tomorrow", "kal") and timestamps are deterministic in tests. */
interface Clock {
    fun now(): Instant
    fun zone(): ZoneId
    fun today(): LocalDate = now().atZone(zone()).toLocalDate()

    object System : Clock {
        override fun now(): Instant = Instant.now()
        override fun zone(): ZoneId = ZoneId.systemDefault()
    }

    class Fixed(private val instant: Instant, private val zoneId: ZoneId = ZoneId.of("Asia/Kolkata")) : Clock {
        override fun now(): Instant = instant
        override fun zone(): ZoneId = zoneId
    }
}

/** Wall-clock stopwatch for stage timings. */
class Stopwatch {
    private val start = java.lang.System.nanoTime()
    fun elapsedMs(): Long = (java.lang.System.nanoTime() - start) / 1_000_000
}
