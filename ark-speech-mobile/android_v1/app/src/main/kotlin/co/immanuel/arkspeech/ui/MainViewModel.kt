package co.immanuel.arkspeech.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import co.immanuel.arkspeech.app.ArkApp
import co.immanuel.arkspeech.capture.PushToTalkRecorder
import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.audio.UnsupportedAudioFormatException
import co.immanuel.arkspeech.core.audio.WavCodec
import co.immanuel.arkspeech.core.config.ConfigBundle
import co.immanuel.arkspeech.core.config.ConfigJson
import co.immanuel.arkspeech.core.config.DefinitionValidator
import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.config.InvalidDefinitionException
import co.immanuel.arkspeech.core.model.ModelPack
import co.immanuel.arkspeech.core.nlu.ExtractionEngine
import co.immanuel.arkspeech.core.nlu.ExtractionResult
import co.immanuel.arkspeech.core.nlu.IntentClassification
import co.immanuel.arkspeech.core.nlu.LexicalIntentClassifier
import co.immanuel.arkspeech.core.nlu.NormalizedText
import co.immanuel.arkspeech.core.pipeline.Consent
import co.immanuel.arkspeech.core.pipeline.PipelineResult
import co.immanuel.arkspeech.core.pipeline.PipelineRun
import co.immanuel.arkspeech.core.pipeline.PipelineStatus
import co.immanuel.arkspeech.model.DownloadProgress
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class Screen { RECORD, MODELS, ADMIN, SETTINGS }

/** Where the current interaction is. */
enum class Phase { IDLE, RECORDING, PROCESSING, REVIEW, RESULT, ERROR }

data class UiState(
    val screen: Screen = Screen.RECORD,
    val phase: Phase = Phase.IDLE,
    val recorder: PushToTalkRecorder.State = PushToTalkRecorder.State.Idle,
    val processingLabel: String = "",
    val result: PipelineResult? = null,
    val transcriptDraft: String = "",
    val languageChoice: String? = null,
    val consent: Consent = Consent(),
    val error: String? = null,
    val busy: Boolean = false,
    val message: String? = null,
    val downloads: Map<String, DownloadProgress> = emptyMap(),
    val modelsVersion: Int = 0,
    val adminUnlocked: Boolean = false,
    val editing: ExtractionDefinition? = null,
    val editorErrors: List<String> = emptyList(),
    val testOutput: String = "",
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val graph = (app as ArkApp).graph
    val recorder = PushToTalkRecorder()

    private val _state = MutableStateFlow(UiState(consent = Consent(graph.settings.defaultRetainAudio, graph.settings.defaultRetainTranscript)))
    val state: StateFlow<UiState> = _state

    private var run: PipelineRun? = null
    private var lastAudio: PcmAudio? = null
    private var downloadWatch: Job? = null

    val settings get() = graph.settings
    val store get() = graph.store
    val catalog get() = graph.catalog
    val repository get() = graph.repository
    val retention get() = graph.retention
    val deviceInfo get() = graph.deviceInfo
    fun engineSummary() = graph.engineSummary()

    init {
        viewModelScope.launch { recorder.state.collect { rs -> onRecorderState(rs) } }
        refreshDownloads()
    }

    fun navigate(screen: Screen) = _state.update { it.copy(screen = screen, message = null) }
    fun dismissMessage() = _state.update { it.copy(message = null) }

    // ---------- recording ----------

    fun startRecording() {
        if (recorder.isRecording) return
        run = null; lastAudio = null
        _state.update { it.copy(phase = Phase.RECORDING, result = null, error = null, transcriptDraft = "", languageChoice = null) }
        recorder.start(viewModelScope)
    }

    fun stopRecording() = recorder.stop()

    fun cancelRecording() {
        recorder.cancel()
        _state.update { it.copy(phase = Phase.IDLE) }
    }

    fun discard() {
        run = null; lastAudio = null
        recorder.reset()
        _state.update { it.copy(phase = Phase.IDLE, result = null, error = null, transcriptDraft = "", languageChoice = null, recorder = PushToTalkRecorder.State.Idle) }
    }

    private fun onRecorderState(rs: PushToTalkRecorder.State) {
        _state.update { it.copy(recorder = rs) }
        when (rs) {
            is PushToTalkRecorder.State.Finished -> process(rs.audio)
            is PushToTalkRecorder.State.Error -> _state.update { it.copy(phase = Phase.ERROR, error = rs.message) }
            is PushToTalkRecorder.State.Cancelled -> _state.update { it.copy(phase = Phase.IDLE) }
            else -> {}
        }
    }

    /** A WAV picked from storage (for testing with files) goes through exactly the same pipeline. */
    fun processWav(uri: Uri) {
        viewModelScope.launch {
            try {
                val bytes = withContext(Dispatchers.IO) { getApplication<Application>().contentResolver.openInputStream(uri)!!.use { it.readBytes() } }
                val decoded = withContext(Dispatchers.Default) { WavCodec.decode(bytes) }
                process(decoded.audio)
            } catch (e: UnsupportedAudioFormatException) {
                _state.update { it.copy(phase = Phase.ERROR, error = "Unsupported format: ${e.message}") }
            } catch (e: Exception) {
                _state.update { it.copy(phase = Phase.ERROR, error = "Could not read file: ${e.javaClass.simpleName}") }
            }
        }
    }

    private fun process(audio: PcmAudio) {
        lastAudio = audio
        _state.update { it.copy(phase = Phase.PROCESSING, processingLabel = "Checking audio…", busy = true) }
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val pipeline = graph.pipeline()
                val r = pipeline.begin(audio, _state.value.consent)
                if (r.isTerminal) { publish(pipeline, r); return@launch }
                label("Identifying language…")
                pipeline.detectLanguage(r)
                if (r.status == PipelineStatus.UNSUPPORTED_LANGUAGE) { publish(pipeline, r); return@launch }
                label("Transcribing…")
                pipeline.transcribe(r)
                run = r
                val result = pipeline.result(r)
                _state.update { it.copy(phase = if (r.isTerminal) Phase.RESULT else Phase.REVIEW, result = result, transcriptDraft = result.transcript?.text ?: "", languageChoice = result.language?.code, busy = false) }
            } catch (e: Exception) {
                graph.log.error("MainViewModel", "processing failed", emptyMap(), e)
                _state.update { it.copy(phase = Phase.ERROR, error = "Processing failed: ${e.javaClass.simpleName}${e.message?.let { m -> ": ${m.take(160)}" } ?: ""}", busy = false) }
            }
        }
    }

    private fun label(s: String) = _state.update { it.copy(processingLabel = s) }

    private fun publish(pipeline: co.immanuel.arkspeech.core.pipeline.SpeechPipeline, r: PipelineRun) {
        run = r
        val result = pipeline.result(r)
        _state.update { it.copy(phase = Phase.RESULT, result = result, busy = false) }
    }

    // ---------- review ----------

    fun setTranscriptDraft(text: String) = _state.update { it.copy(transcriptDraft = text) }

    fun chooseLanguage(code: String) = _state.update { it.copy(languageChoice = code) }

    /** Re-run ASR in the chosen language (user override). */
    fun retranscribe() {
        val r = run ?: return
        val code = _state.value.languageChoice ?: return
        _state.update { it.copy(phase = Phase.PROCESSING, processingLabel = "Transcribing in ${co.immanuel.arkspeech.core.lang.Languages.displayName(code)}…", busy = true) }
        viewModelScope.launch(Dispatchers.Default) {
            val pipeline = graph.pipeline()
            pipeline.overrideLanguage(r, code)
            pipeline.transcribe(r)
            val result = pipeline.result(r)
            _state.update { it.copy(phase = if (r.isTerminal) Phase.RESULT else Phase.REVIEW, result = result, transcriptDraft = result.transcript?.text ?: "", busy = false) }
        }
    }

    /** Accept the (possibly edited) transcript and extract intent and fields. */
    fun extract(forcedDefinitionId: String? = null) {
        val r = run ?: return
        val draft = _state.value.transcriptDraft
        val code = _state.value.languageChoice
        _state.update { it.copy(phase = Phase.PROCESSING, processingLabel = "Understanding…", busy = true) }
        viewModelScope.launch(Dispatchers.Default) {
            val pipeline = graph.pipeline()
            if (code != null && code != r.effectiveLanguage) pipeline.overrideLanguage(r, code)
            else if (r.language?.uncertain == true && code != null) pipeline.overrideLanguage(r, code) // user confirmed the guess
            if (draft != (r.transcript?.text ?: "")) pipeline.editTranscript(r, draft)
            pipeline.extract(r, forcedDefinitionId)
            finish(pipeline, r)
        }
    }

    fun provideField(field: String, value: String) {
        val r = run ?: return
        viewModelScope.launch(Dispatchers.Default) { val p = graph.pipeline(); p.provideFieldValue(r, field, value); finish(p, r) }
    }

    fun confirmField(field: String) {
        val r = run ?: return
        viewModelScope.launch(Dispatchers.Default) { val p = graph.pipeline(); p.confirmFieldValue(r, field); finish(p, r) }
    }

    fun chooseIntent(definitionId: String) = extract(definitionId)

    private fun finish(pipeline: co.immanuel.arkspeech.core.pipeline.SpeechPipeline, r: PipelineRun) {
        val result = pipeline.result(r)
        retain(result)
        _state.update { it.copy(phase = Phase.RESULT, result = result, busy = false) }
    }

    fun setConsent(audio: Boolean? = null, transcript: Boolean? = null) {
        _state.update { it.copy(consent = Consent(audio ?: it.consent.retainAudio, transcript ?: it.consent.retainTranscript)) }
    }

    /** Writes audio/transcript only when the user ticked the boxes, and marks the result accordingly. */
    private fun retain(result: PipelineResult) {
        val c = _state.value.consent
        if (c.retainAudio) lastAudio?.let { graph.retention.saveAudio(result.requestId, it) }
        if (c.retainTranscript) graph.retention.saveResult(result.copy(retention = result.retention.copy(consentToRetainAudio = c.retainAudio, consentToRetainTranscript = true, audioRetained = c.retainAudio, transcriptRetained = true)))
    }

    fun backToReview() = _state.update { it.copy(phase = Phase.REVIEW) }

    // ---------- models ----------

    fun download(pack: ModelPack) {
        graph.downloader.enqueue(pack, graph.settings.wifiOnlyDownloads)
        refreshDownloads()
    }

    fun cancelDownload(pack: ModelPack) { graph.downloader.cancel(pack); refreshDownloads() }

    fun deletePack(pack: ModelPack) {
        graph.releaseEngines()
        graph.store.delete(pack)
        _state.update { it.copy(modelsVersion = it.modelsVersion + 1, message = "${pack.displayName} removed") }
    }

    fun verifyPack(pack: ModelPack) {
        viewModelScope.launch {
            _state.update { it.copy(busy = true) }
            val r = graph.downloader.verify(pack)
            val bad = r.filterValues { !it }.keys
            _state.update { it.copy(busy = false, message = if (bad.isEmpty()) "${pack.displayName}: all files verified" else "${pack.displayName}: checksum failed for ${bad.joinToString()}") }
        }
    }

    fun refreshDownloads() {
        downloadWatch?.cancel()
        downloadWatch = viewModelScope.launch {
            graph.downloader.watch(graph.catalog.packs).collect { progress ->
                _state.update { it.copy(downloads = progress) }
            }
            graph.invalidateEngines()
            _state.update { it.copy(modelsVersion = it.modelsVersion + 1) }
        }
    }

    // ---------- admin ----------

    fun unlockAdmin(pin: String): Boolean {
        val ok = graph.settings.checkAdminPin(pin)
        _state.update { it.copy(adminUnlocked = ok) }
        return ok
    }

    fun lockAdmin() = _state.update { it.copy(adminUnlocked = false, editing = null) }

    fun definitions(): List<ExtractionDefinition> = graph.repository.list()

    fun setDefinitionEnabled(id: String, enabled: Boolean) {
        graph.repository.setEnabled(id, enabled, author = "admin")
        _state.update { it.copy(modelsVersion = it.modelsVersion + 1) }
    }

    fun startEditing(def: ExtractionDefinition?) = _state.update { it.copy(editing = def ?: newDefinition(), editorErrors = emptyList(), testOutput = "") }

    fun stopEditing() = _state.update { it.copy(editing = null, editorErrors = emptyList(), testOutput = "") }

    fun validateDefinitionJson(json: String): ExtractionDefinition? {
        return try {
            val def = ConfigJson.pretty.decodeFromString(ExtractionDefinition.serializer(), json)
            val errors = DefinitionValidator.validate(def)
            _state.update { it.copy(editorErrors = errors.map { e -> "${e.path}: ${e.message}" }) }
            if (errors.isEmpty()) def else null
        } catch (e: Exception) {
            _state.update { it.copy(editorErrors = listOf("JSON: ${e.message?.take(300)}")) }
            null
        }
    }

    fun saveDefinition(def: ExtractionDefinition, note: String): Boolean = try {
        graph.repository.save(def, author = "admin", note = note)
        graph.invalidateEngines()
        _state.update { it.copy(editing = null, editorErrors = emptyList(), message = "Saved ${def.id} as a new version", modelsVersion = it.modelsVersion + 1) }
        true
    } catch (e: InvalidDefinitionException) {
        _state.update { it.copy(editorErrors = e.errors.map { c -> "${c.path}: ${c.message}" }) }
        false
    }

    fun rollback(id: String, version: Int) {
        graph.repository.rollback(id, version, author = "admin")
        _state.update { it.copy(message = "Rolled back $id to version $version", modelsVersion = it.modelsVersion + 1) }
    }

    fun deleteDefinition(id: String) {
        graph.repository.delete(id)
        _state.update { it.copy(message = "Deleted $id", modelsVersion = it.modelsVersion + 1) }
    }

    /** Admin "try it": classify and extract a typed sentence against one definition. */
    fun testDefinition(def: ExtractionDefinition, text: String, language: String) {
        val nt = NormalizedText(text)
        val cls: IntentClassification = LexicalIntentClassifier().classify(nt, language, listOf(def))
        val ext: ExtractionResult = ExtractionEngine().extract(nt, def, language)
        val sb = StringBuilder()
        sb.append("intent: ${cls.status} score=${"%.2f".format(cls.confidence)} phrases=${cls.best?.matchedPhrases ?: emptyList<String>()}\n")
        for (f in ext.fields.values) sb.append("${f.name}: ${f.status} value=${f.value ?: "-"} raw=${f.raw ?: "-"} conf=${"%.2f".format(f.confidence)}${f.reason?.let { " ($it)" } ?: ""}\n")
        _state.update { it.copy(testOutput = sb.toString()) }
    }

    fun exportBundle(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val json = ConfigJson.pretty.encodeToString(ConfigBundle.serializer(), graph.repository.exportBundle("Exported from Ark Speech"))
            getApplication<Application>().contentResolver.openOutputStream(uri)?.use { it.write(json.toByteArray()) }
            _state.update { it.copy(message = "Configuration exported") }
        }
    }

    fun importBundle(uri: Uri, replace: Boolean) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val text = getApplication<Application>().contentResolver.openInputStream(uri)!!.use { it.readBytes().toString(Charsets.UTF_8) }
                val bundle = try { ConfigJson.pretty.decodeFromString(ConfigBundle.serializer(), text) } catch (e: Exception) {
                    // A single definition file is accepted too.
                    ConfigBundle(definitions = listOf(ConfigJson.pretty.decodeFromString(ExtractionDefinition.serializer(), text)))
                }
                val report = graph.repository.importBundle(bundle, author = "admin", replace = replace)
                graph.invalidateEngines()
                _state.update { it.copy(message = if (report.ok) "Imported ${report.imported.size} definition(s)" else "Import rejected: " + report.errors.joinToString("; ") { e -> "${e.path}: ${e.message}" }.take(400), modelsVersion = it.modelsVersion + 1) }
            } catch (e: Exception) {
                _state.update { it.copy(message = "Import failed: ${e.message?.take(200)}") }
            }
        }
    }

    fun saveRouting(table: RoutingTable) { graph.repository.saveRouting(table); graph.invalidateEngines(); _state.update { it.copy(modelsVersion = it.modelsVersion + 1, message = "Routing saved") } }

    fun bumpVersion() = _state.update { it.copy(modelsVersion = it.modelsVersion + 1) }

    private fun newDefinition() = ExtractionDefinition(
        id = "new_intent",
        intent = co.immanuel.arkspeech.core.config.IntentSpec("new_intent", "Describe the intent"),
        languages = listOf("en", "hi"),
        triggers = co.immanuel.arkspeech.core.config.IntentTriggers(phrases = mapOf("en" to listOf("example phrase"), "hi" to listOf("उदाहरण"))),
        fields = listOf(co.immanuel.arkspeech.core.config.FieldDefinition("example_date", "An example date", co.immanuel.arkspeech.core.config.FieldType.DATE, required = true)),
    )

    override fun onCleared() {
        super.onCleared()
        if (recorder.isRecording) recorder.cancel()
    }
}
