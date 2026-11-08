package co.immanuel.arkspeech.app

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/** Static device facts attached to every result's `processing.device` and to benchmark output. */
object DeviceInfo {
    fun collect(context: Context): Map<String, String> {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        return linkedMapOf(
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "sdk" to Build.VERSION.SDK_INT.toString(),
            "abi" to (Build.SUPPORTED_ABIS.firstOrNull() ?: "?"),
            "cores" to Runtime.getRuntime().availableProcessors().toString(),
            "totalRamMb" to (mi.totalMem / (1024 * 1024)).toString(),
            "availRamMb" to (mi.availMem / (1024 * 1024)).toString(),
            "lowRamDevice" to am.isLowRamDevice.toString(),
            "memoryClassMb" to am.memoryClass.toString(),
            "largeMemoryClassMb" to am.largeMemoryClass.toString(),
            "emulator" to isEmulator().toString(),
        )
    }

    /** Android emulator images (goldfish/ranchu/cuttlefish) pass the host CPU's feature flags through. */
    fun isEmulator(): Boolean = Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish") || Build.HARDWARE.contains("cutf") ||
        Build.FINGERPRINT.contains("generic") || Build.FINGERPRINT.contains("emulator") || Build.MODEL.contains("sdk_gphone") || Build.PRODUCT.startsWith("sdk")

    /** Inference threads: big/medium cores help, but more than 4 threads rarely does on phones. */
    fun inferenceThreads(): Int = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
}
