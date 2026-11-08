package co.immanuel.arkspeech.app

import android.app.Application

class ArkApp : Application() {
    lateinit var graph: AppGraph
        private set

    override fun onCreate() {
        super.onCreate()
        graph = AppGraph(this)
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // Drop resident models when the system is under pressure; they reload on next use.
        if (level >= TRIM_MEMORY_RUNNING_LOW) graph.releaseEngines()
    }
}
