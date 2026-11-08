package co.immanuel.arkspeech.app

import android.content.Context
import co.immanuel.arkspeech.core.asr.RecognizerFactory
import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.asr.SpeechRecognizer
import co.immanuel.arkspeech.core.asr.Transcript
import co.immanuel.arkspeech.core.audio.AudioValidationPolicy
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.config.FileDefinitionRepository
import co.immanuel.arkspeech.core.lang.LanguageDetector
import co.immanuel.arkspeech.core.lang.LanguagePolicy
import co.immanuel.arkspeech.core.log.SafeLog
import co.immanuel.arkspeech.core.model.ModelKind
import co.immanuel.arkspeech.core.model.ModelPack
import co.immanuel.arkspeech.core.pipeline.PipelineDependencies
import co.immanuel.arkspeech.core.pipeline.SpeechPipeline
import co.immanuel.arkspeech.core.vad.EnergyVad
import co.immanuel.arkspeech.core.vad.SpeechSegment
import co.immanuel.arkspeech.core.vad.VoiceActivityDetector
import co.immanuel.arkspeech.engine.AnuvaniLanguageDetector
import co.immanuel.arkspeech.engine.SherpaRecognizerFactory
import co.immanuel.arkspeech.engine.SherpaVad
import co.immanuel.arkspeech.engine.WhisperLanguageDetector
import co.immanuel.arkspeech.model.ModelDownloader
import co.immanuel.arkspeech.model.ModelStore
import java.io.File

/** Which engine each stage will use right now, for the Models screen and the result's processing block. */
data class EngineSummary(val vad: String, val lid: String?, val asrDefault: String?, val asrInstalled: List<String>)

/**
 * Hand-wired dependency graph (no DI framework). Engines are created lazily from whatever packs are
 * installed and rebuilt when the model set or settings change. Everything speech-related is
 * released by [releaseEngines].
 */
class AppGraph(private val context: Context) {

    val log: SafeLog = AndroidSafeLog()
    val settings = Settings(context)
    val catalog = ModelStore.loadCatalog(context)
    val store = ModelStore(context, catalog)
    val downloader = ModelDownloader(context, store)
    val retention = Retention(context)
    val repository = FileDefinitionRepository(File(context.filesDir, "config"))
    val deviceInfo: Map<String, String> by lazy { DeviceInfo.collect(context) }

    private var vad: VoiceActivityDetector? = null
    private var lid: LanguageDetector? = null
    private var factory: CachingRecognizerFactory? = null
    private var pipeline: SpeechPipeline? = null
    private var engineFingerprint: String? = null

    init {
        seedIfNeeded()
    }

    /** First launch: load the sample definitions and default routing shipped in assets. */
    private fun seedIfNeeded() {
        if (settings.seeded && repository.list().isNotEmpty()) return
        val names = context.assets.list("definitions")?.filter { it.endsWith(".json") } ?: emptyList()
        for (n in names) {
            runCatching {
                val def = ConfigJson.pretty.decodeFromString(ExtractionDefinition.serializer(), context.assets.open("definitions/$n").bufferedReader().readText())
                if (repository.get(def.id) == null) repository.save(def, author = "seed", note = "Seeded from bundled sample")
            }.onFailure { log.warn("AppGraph", "seed failed", mapOf("file" to n), it) }
        }
        if (repository.routing() == null) {
            runCatching { ConfigJson.pretty.decodeFromString(RoutingTable.serializer(), context.assets.open("routing.json").bufferedReader().readText()) }
                .getOrNull()?.let { repository.saveRouting(it) }
        }
        settings.seeded = true
    }

    val routing: RoutingTable get() = repository.routing() ?: RoutingTable(catalog.byKind(ModelKind.ASR).firstOrNull { it.tier == "default" }?.id ?: "")

    @Synchronized
    fun pipeline(): SpeechPipeline {
        val fp = fingerprint()
        if (pipeline == null || fp != engineFingerprint) {
            releaseEngines()
            val threads = settings.inferenceThreads
            vad = installedPack(ModelKind.VAD)?.let { p -> SherpaVad(store.file(p, p.files.first().name), threads = 1) } ?: EnergyVad()
            lid = buildLid(threads)
            factory = CachingRecognizerFactory(SherpaRecognizerFactory(store, threads), settings.keepModelsLoaded)
            pipeline = SpeechPipeline(
                PipelineDependencies(
                    vad = vad!!, languageDetector = lid, recognizers = factory!!, routing = routing, definitions = repository,
                    audioPolicy = repository.audioPolicy() ?: AudioValidationPolicy(),
                    languagePolicy = repository.languagePolicy() ?: LanguagePolicy(),
                    log = log, deviceInfo = deviceInfo,
                ),
            )
            engineFingerprint = fp
        }
        return pipeline!!
    }

    private fun buildLid(threads: Int): LanguageDetector? {
        installedPack(ModelKind.LID)?.let { p -> return AnuvaniLanguageDetector(store.file(p, p.files.first().name), threads, disableKleidiAi = settings.disableAcceleratedKernels) }
        // Fallback: a Whisper pack that declares lid=true (tiny/base).
        val w = store.installed(ModelKind.ASR).firstOrNull { it.params["lid"] == "true" } ?: return null
        return WhisperLanguageDetector(store.file(w, w.params.getValue("encoder")), store.file(w, w.params.getValue("decoder")), threads)
    }

    private fun installedPack(kind: ModelKind): ModelPack? = store.installed(kind).sortedBy { if (it.tier == "default") 0 else 1 }.firstOrNull()

    fun engineSummary(): EngineSummary {
        val vadName = if (installedPack(ModelKind.VAD) != null) "Silero VAD (sherpa-onnx)" else "Energy VAD (fallback: install Silero)"
        val lidName = installedPack(ModelKind.LID)?.displayName ?: store.installed(ModelKind.ASR).firstOrNull { it.params["lid"] == "true" }?.let { "${it.displayName} (vote-based fallback)" }
        val installed = store.installed(ModelKind.ASR).map { it.id }
        val def = routing.candidates(null).firstOrNull { it in installed }
        return EngineSummary(vadName, lidName, def, installed)
    }

    /** Model installs/deletes or settings changes invalidate the engine set. */
    private fun fingerprint(): String = buildString {
        append(settings.inferenceThreads).append('|').append(settings.keepModelsLoaded).append('|').append(settings.disableAcceleratedKernels).append('|')
        catalog.packs.forEach { append(it.id).append('=').append(store.state(it)).append(',') }
        append(repository.routing()?.hashCode())
    }

    @Synchronized
    fun releaseEngines() {
        (vad as? AutoCloseable)?.let { runCatching { it.close() } }
        (lid as? AutoCloseable)?.let { runCatching { it.close() } }
        factory?.releaseAll()
        vad = null; lid = null; factory = null; pipeline = null; engineFingerprint = null
    }

    /** Forces engines to be rebuilt on next use (after a model download finishes, for example). */
    @Synchronized
    fun invalidateEngines() { engineFingerprint = null }
}

/**
 * Keeps the last ASR model resident so consecutive recordings do not pay the multi-second load again.
 * The pipeline calls close() on the handle it receives; the handle forwards that to nothing when
 * [keepLoaded], otherwise it releases the model.
 */
class CachingRecognizerFactory(private val inner: RecognizerFactory, private val keepLoaded: Boolean) : RecognizerFactory {
    private var current: SpeechRecognizer? = null
    private var currentId: String? = null

    override fun availableModelIds(): Set<String> = inner.availableModelIds()

    @Synchronized
    override fun create(modelId: String): SpeechRecognizer {
        if (currentId != modelId) { releaseAll(); current = inner.create(modelId); currentId = modelId }
        val rec = current!!
        return object : SpeechRecognizer {
            override val modelId = rec.modelId
            override val engineId = rec.engineId
            override val languages = rec.languages
            override suspend fun transcribe(audio: PcmAudio, segments: List<SpeechSegment>, language: String?): Transcript = rec.transcribe(audio, segments, language)
            override fun close() { if (!keepLoaded) releaseAll() }
        }
    }

    @Synchronized
    fun releaseAll() { current?.let { runCatching { it.close() } }; current = null; currentId = null }
}
