package co.immanuel.arkspeech.app

import android.content.Context
import android.content.SharedPreferences
import java.security.MessageDigest

/** Small user/admin preferences. Nothing here is sensitive except the admin PIN, which is stored hashed. */
class Settings(context: Context) {
    private val p: SharedPreferences = context.getSharedPreferences("ark_settings", Context.MODE_PRIVATE)

    var wifiOnlyDownloads: Boolean
        get() = p.getBoolean("wifi_only", true)
        set(v) = p.edit().putBoolean("wifi_only", v).apply()

    var inferenceThreads: Int
        get() = p.getInt("threads", DeviceInfo.inferenceThreads())
        set(v) = p.edit().putInt("threads", v.coerceIn(1, 8)).apply()

    /** Keep the ASR model resident between recordings (faster) or release it (less RAM). */
    var keepModelsLoaded: Boolean
        get() = p.getBoolean("keep_loaded", true)
        set(v) = p.edit().putBoolean("keep_loaded", v).apply()

    var defaultRetainAudio: Boolean
        get() = p.getBoolean("retain_audio", false)
        set(v) = p.edit().putBoolean("retain_audio", v).apply()

    var defaultRetainTranscript: Boolean
        get() = p.getBoolean("retain_transcript", false)
        set(v) = p.edit().putBoolean("retain_transcript", v).apply()

    /** Disable ONNX Runtime's KleidiAI kernels (see AnuvaniLanguageDetector). Default: on emulators only. */
    var disableAcceleratedKernels: Boolean
        get() = p.getBoolean("disable_kleidiai", DeviceInfo.isEmulator())
        set(v) = p.edit().putBoolean("disable_kleidiai", v).apply()

    var holdToTalk: Boolean
        get() = p.getBoolean("hold_to_talk", true)
        set(v) = p.edit().putBoolean("hold_to_talk", v).apply()

    var seeded: Boolean
        get() = p.getBoolean("seeded", false)
        set(v) = p.edit().putBoolean("seeded", v).apply()

    val hasAdminPin: Boolean get() = p.getString("admin_pin_sha", null) != null

    fun setAdminPin(pin: String?) {
        if (pin.isNullOrBlank()) p.edit().remove("admin_pin_sha").apply() else p.edit().putString("admin_pin_sha", sha(pin)).apply()
    }

    fun checkAdminPin(pin: String): Boolean = p.getString("admin_pin_sha", null)?.equals(sha(pin)) ?: true

    private fun sha(s: String) = MessageDigest.getInstance("SHA-256").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }
}
