package co.immanuel.arkspeech.app

import android.util.Log
import co.immanuel.arkspeech.core.log.Redactor
import co.immanuel.arkspeech.core.log.SafeLog

/** Logcat sink. Metadata goes through [Redactor]; lines are capped so nothing large leaks by accident. */
class AndroidSafeLog(private val minLevel: Int = Log.DEBUG) : SafeLog {
    private fun line(message: String, meta: Map<String, Any?>) = Redactor.format(message, meta).take(MAX_CHARS)

    override fun debug(tag: String, message: String, meta: Map<String, Any?>) { if (minLevel <= Log.DEBUG) Log.d(prefix(tag), line(message, meta)) }
    override fun info(tag: String, message: String, meta: Map<String, Any?>) { if (minLevel <= Log.INFO) Log.i(prefix(tag), line(message, meta)) }
    override fun warn(tag: String, message: String, meta: Map<String, Any?>, error: Throwable?) { Log.w(prefix(tag), line(message, meta) + (error?.let { " ${it.javaClass.simpleName}" } ?: "")) }
    override fun error(tag: String, message: String, meta: Map<String, Any?>, error: Throwable?) { Log.e(prefix(tag), line(message, meta) + (error?.let { " ${it.javaClass.simpleName}: ${it.message?.take(120)}" } ?: "")) }

    private fun prefix(tag: String) = "Ark/$tag".take(23)

    companion object { const val MAX_CHARS = 400 }
}
