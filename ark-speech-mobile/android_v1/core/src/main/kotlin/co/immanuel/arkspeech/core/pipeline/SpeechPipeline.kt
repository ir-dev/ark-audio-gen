package co.immanuel.arkspeech.core.pipeline

import co.immanuel.arkspeech.core.asr.ModelRouter
import co.immanuel.arkspeech.core.asr.NoModelAvailableException
import co.immanuel.arkspeech.core.asr.RecognizerFactory
import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.asr.Transcript
import co.immanuel.arkspeech.core.asr.TranscriptQualityEstimator
import co.immanuel.arkspeech.core.audio.AudioQualityReport
import co.immanuel.arkspeech.core.audio.AudioValidationPolicy
import co.immanuel.arkspeech.core.audio.AudioValidator
import co.immanuel.arkspeech.core.audio.AudioVerdict
import co.immanuel.arkspeech.core.audio.PcmAudio
import co.immanuel.arkspeech.core.config.DefinitionRepository
import co.immanuel.arkspeech.core.config.ExtractionDefinition
import co.immanuel.arkspeech.core.config.FallbackBehaviour
import co.immanuel.arkspeech.core.lang.LanguageDetection
import co.immanuel.arkspeech.core.lang.LanguageDetector
import co.immanuel.arkspeech.core.lang.LanguagePolicy
import co.immanuel.arkspeech.core.lang.Languages
import co.immanuel.arkspeech.core.lang.SegmentedLanguageIdentifier
import co.immanuel.arkspeech.core.log.SafeLog
import co.immanuel.arkspeech.core.nlu.ExtractionEngine
import co.immanuel.arkspeech.core.nlu.ExtractionResult
import co.immanuel.arkspeech.core.nlu.FieldStatus
import co.immanuel.arkspeech.core.nlu.IntentCandidate
import co.immanuel.arkspeech.core.nlu.IntentClassification
import co.immanuel.arkspeech.core.nlu.IntentClassifier
import co.immanuel.arkspeech.core.nlu.IntentStatus
import co.immanuel.arkspeech.core.nlu.LexicalIntentClassifier
import co.immanuel.arkspeech.core.nlu.NormalizedText
import co.immanuel.arkspeech.core.util.Clock
import co.immanuel.arkspeech.core.util.Stopwatch
import co.immanuel.arkspeech.core.vad.SpeechSegment
import co.immanuel.arkspeech.core.vad.VoiceActivityDetector
import java.time.format.DateTimeFormatter
import java.util.UUID

/** Everything the pipeline needs; the app wires real engines, tests wire fakes. */
class PipelineDependencies(
    val vad: VoiceActivityDetector,
    val languageDetector: LanguageDetector?,
    val recognizers: RecognizerFactory,
    val routing: RoutingTable,
    val definitions: DefinitionRepository,
    val audioPolicy: AudioValidationPolicy = AudioValidationPolicy(),
    val languagePolicy: LanguagePolicy = LanguagePolicy(),
    val intentClassifier: IntentClassifier = LexicalIntentClassifier(),
    val clock: Clock = Clock.System,
    val log: SafeLog = SafeLog.Noop,
    val deviceInfo: Map<String, String> = emptyMap(),
)

/** The user's consent choices for this interaction. */
data class Consent(val retainAudio: Boolean = false, val retainTranscript: Boolean = false)

/**
 * Mutable state of one interaction as it moves through the stages. The UI drives it step by step
 * (so it can pause for language override, transcript review and field clarification) and the one-shot
 * [SpeechPipeline.runAll] drives it straight through. [toResult] renders the output contract at any point.
 */
class PipelineRun internal constructor(val requestId: String, internal val startedAt: java.time.Instant, val consent: Consent) {
    var audio: PcmAudio? = null
        internal set
    var quality: AudioQualityReport? = null
        internal set
    var segments: List<SpeechSegment> = emptyList()
        internal set
    var language: LanguageDetection? = null
        internal set
    var transcript: Transcript? = null
        internal set
    var intent: IntentClassification? = null
        internal set
    var extraction: ExtractionResult? = null
        internal set
    var definition: ExtractionDefinition? = null
        internal set
    var status: PipelineStatus = PipelineStatus.IN_PROGRESS
        internal set
    internal val stages = ArrayList<StageTiming>()
    internal val warnings = ArrayList<String>()
    internal var vadEngine: String? = null
    internal var lidEngine: String? = null
    internal var asrEngine: String? = null
    internal var asrModelId: String? = null
    internal var nluEngine: String? = null
    internal var audioRetained = false
    internal var transcriptRetained = false

    val effectiveLanguage: String? get() = language?.code
    val isTerminal: Boolean get() = status != PipelineStatus.IN_PROGRESS && status != PipelineStatus.NEEDS_CLARIFICATION

    internal fun stage(name: String, ms: Long, status: String, detail: String? = null) { stages += StageTiming(name, ms, status, detail) }
    internal fun warn(message: String) { warnings += message }

    fun toResult(deviceInfo: Map<String, String> = emptyMap(), now: java.time.Instant = java.time.Instant.now()): PipelineResult {
        val clarifications = buildClarifications()
        val ext = extraction
        return PipelineResult(
            requestId = requestId,
            status = status,
            audio = quality,
            speechSegments = segments,
            language = language,
            transcript = transcript,
            intent = intent,
            fields = ext?.fields ?: emptyMap(),
            missingRequired = ext?.missingRequired ?: emptyList(),
            clarifications = clarifications,
            warnings = warnings.toList(),
            processing = ProcessingInfo(
                startedAt = DateTimeFormatter.ISO_INSTANT.format(startedAt),
                totalMs = java.time.Duration.between(startedAt, now).toMillis(),
                stages = stages.toList(),
                vadEngine = vadEngine, lidEngine = lidEngine, asrEngine = asrEngine, asrModelId = asrModelId, nluEngine = nluEngine,
                device = deviceInfo,
            ),
            retention = RetentionInfo(consent.retainAudio, consent.retainTranscript, audioRetained, transcriptRetained, true),
        )
    }

    private fun buildClarifications(): List<Clarification> {
        val out = ArrayList<Clarification>()
        language?.let { l ->
            if (l.uncertain && l.source == co.immanuel.arkspeech.core.lang.LanguageSource.AUTO) {
                val options = (listOfNotNull(l.code) + l.alternatives.map { it.code }).distinct().take(5)
                out += Clarification(ClarificationKind.LANGUAGE, null, if (l.isCodeMixed) "The recording mixes languages. Which language should be used for transcription?" else "The language could not be identified confidently. Which language was spoken?", options)
            }
        }
        intent?.let { i ->
            when (i.status) {
                IntentStatus.AMBIGUOUS -> out += Clarification(ClarificationKind.INTENT, null, "Which of these did you mean?", (listOfNotNull(i.best) + i.alternatives).map { it.definitionId })
                IntentStatus.LOW_CONFIDENCE -> out += Clarification(ClarificationKind.INTENT, null, "Is this what you meant?", listOfNotNull(i.best?.definitionId) + i.alternatives.map { it.definitionId })
                else -> {}
            }
        }
        extraction?.fields?.values?.forEach { f ->
            if (!f.needsUser) return@forEach
            val def = definition?.fields?.firstOrNull { it.name == f.name }
            val label = def?.description?.ifBlank { f.name } ?: f.name
            val enumOptions = def?.validation?.allowedValues?.map { it.value } ?: emptyList()
            out += when (f.status) {
                FieldStatus.MISSING -> Clarification(ClarificationKind.FIELD_MISSING, f.name, "Please provide: $label", enumOptions)
                FieldStatus.AMBIGUOUS -> Clarification(ClarificationKind.FIELD_AMBIGUOUS, f.name, "Which value is right for $label?", f.candidates.map { it.value }, f.candidates)
                FieldStatus.LOW_CONFIDENCE -> Clarification(ClarificationKind.FIELD_LOW_CONFIDENCE, f.name, "Is '${f.raw ?: f.value}' correct for $label?", listOfNotNull(f.value) + enumOptions, f.candidates)
                FieldStatus.INVALID -> Clarification(ClarificationKind.FIELD_INVALID, f.name, "'${f.raw ?: f.value}' is not valid for $label (${f.reason}). Please correct it.", enumOptions, f.candidates)
                else -> return@forEach
            }
        }
        return out
    }
}

/**
 * Orchestrates: validation → VAD → language identification → model routing → ASR → intent → extraction.
 * Each stage is a plain method so the UI can stop between them; nothing here touches Android.
 */
class SpeechPipeline(private val deps: PipelineDependencies) {

    private val validator = AudioValidator(deps.audioPolicy)
    private val extractor = ExtractionEngine(deps.clock)
    private val router = ModelRouter(deps.routing, deps.recognizers)
    private val log = deps.log

    fun begin(audio: PcmAudio, consent: Consent = Consent()): PipelineRun {
        val run = PipelineRun(UUID.randomUUID().toString(), deps.clock.now(), consent)
        val pcm = audio.resampledTo(PcmAudio.TARGET_SAMPLE_RATE)
        run.audio = pcm
        val sw = Stopwatch()
        val q = validator.validate(pcm)
        run.quality = q
        run.stage("validate", sw.elapsedMs(), q.verdict.name, q.issues.joinToString(",").ifEmpty { null })
        log.info(TAG, "audio validated", mapOf("requestId" to run.requestId, "durationMs" to q.durationMs, "verdict" to q.verdict, "issues" to q.issues))
        if (q.verdict == AudioVerdict.REJECT) { run.status = PipelineStatus.AUDIO_REJECTED; return run }
        if (q.verdict == AudioVerdict.WARN) run.warn("Audio quality issues: ${q.issues.joinToString(", ")}")

        val sw2 = Stopwatch()
        val segs = try { deps.vad.segments(pcm) } catch (e: Exception) { log.warn(TAG, "vad failed", mapOf("requestId" to run.requestId), e); run.warn("VAD failed: ${e.javaClass.simpleName}"); emptyList() }
        run.segments = segs
        run.vadEngine = deps.vad.engineId
        run.stage("vad", sw2.elapsedMs(), if (segs.isEmpty()) "NO_SPEECH" else "OK", "${segs.size} segments, ${segs.sumOf { it.durationMs }} ms speech")
        if (segs.isEmpty()) run.status = PipelineStatus.NO_SPEECH
        return run
    }

    suspend fun detectLanguage(run: PipelineRun): PipelineRun {
        if (run.isTerminal) return run
        val audio = run.audio ?: return run
        val detector = deps.languageDetector
        val sw = Stopwatch()
        if (detector == null) {
            run.language = LanguageDetection(null, 0f, uncertain = true, supported = false, engine = "none")
            run.stage("lid", 0, "SKIPPED", "no language detector installed")
            return run
        }
        val detection = try {
            SegmentedLanguageIdentifier(detector, deps.languagePolicy).identify(audio, run.segments)
        } catch (e: Exception) {
            log.error(TAG, "language identification failed", mapOf("requestId" to run.requestId), e)
            run.warn("Language identification failed: ${e.javaClass.simpleName}")
            LanguageDetection.none(detector.engineId)
        }
        run.language = detection
        run.lidEngine = detector.engineId
        run.stage("lid", sw.elapsedMs(), if (detection.uncertain) "UNCERTAIN" else "OK", detection.code?.let { "$it ${"%.2f".format(detection.confidence)}" })
        log.info(TAG, "language identified", mapOf("requestId" to run.requestId, "code" to detection.code, "confidence" to detection.confidence, "uncertain" to detection.uncertain, "codeMixed" to detection.isCodeMixed))
        if (detection.code != null && !Languages.isSupported(detection.code)) run.status = PipelineStatus.UNSUPPORTED_LANGUAGE
        return run
    }

    /** The user picked a language on the review screen (or the client knows it in advance). */
    fun overrideLanguage(run: PipelineRun, code: String): PipelineRun {
        run.language = LanguageDetection.override(code)
        run.stage("lid-override", 0, "OK", code)
        if (run.status == PipelineStatus.UNSUPPORTED_LANGUAGE) run.status = PipelineStatus.IN_PROGRESS
        return run
    }

    suspend fun transcribe(run: PipelineRun): PipelineRun {
        if (run.isTerminal) return run
        val audio = run.audio ?: return run
        val lang = run.effectiveLanguage
        val sw = Stopwatch()
        val modelId = try { router.resolve(lang) } catch (e: NoModelAvailableException) {
            run.stage("route", sw.elapsedMs(), "NO_MODEL", e.tried.joinToString(","))
            run.warn(e.message ?: "no model")
            run.status = PipelineStatus.MODEL_UNAVAILABLE
            return run
        }
        run.stage("route", sw.elapsedMs(), "OK", modelId)
        val sw2 = Stopwatch()
        try {
            deps.recognizers.create(modelId).use { rec ->
                val raw = rec.transcribe(audio, run.segments, lang)
                val conf = if (raw.confidenceSource == "heuristic") TranscriptQualityEstimator.estimate(raw, run.segments, lang) else raw.confidence
                run.transcript = raw.copy(confidence = conf, language = raw.language ?: lang, modelId = modelId)
                run.asrEngine = rec.engineId
                run.asrModelId = modelId
                run.stage("asr", sw2.elapsedMs(), if (raw.isBlank) "EMPTY" else "OK", "${raw.text.length} chars, conf ${"%.2f".format(conf)}")
                log.info(TAG, "transcribed", mapOf("requestId" to run.requestId, "model" to modelId, "chars" to raw.text.length, "confidence" to conf))
            }
        } catch (e: Exception) {
            log.error(TAG, "asr failed", mapOf("requestId" to run.requestId, "model" to modelId), e)
            run.stage("asr", sw2.elapsedMs(), "FAILED", e.javaClass.simpleName)
            run.warn("Transcription failed: ${e.javaClass.simpleName}")
            run.status = PipelineStatus.FAILED
        }
        if (run.status == PipelineStatus.IN_PROGRESS && run.transcript?.isBlank != false) run.status = PipelineStatus.NO_SPEECH
        return run
    }

    /** The user corrected the transcript on the preview screen. */
    fun editTranscript(run: PipelineRun, text: String): PipelineRun {
        val t = run.transcript
        run.transcript = (t ?: Transcript(text, run.effectiveLanguage)).copy(text = text, edited = true, confidence = 1f, confidenceSource = "user", script = Languages.dominantScript(text))
        run.stage("transcript-edit", 0, "OK")
        // Earlier understanding no longer applies to the new text.
        run.intent = null; run.extraction = null; run.definition = null
        if (text.isNotBlank()) run.status = PipelineStatus.IN_PROGRESS
        return run
    }

    fun extract(run: PipelineRun, forcedDefinitionId: String? = null): PipelineRun {
        if (run.status in BLOCKING) return run
        val transcript = run.transcript ?: return run
        if (transcript.isBlank) { run.status = PipelineStatus.NO_SPEECH; return run }
        val lang = run.effectiveLanguage
        val text = NormalizedText(transcript.text)
        // An uncertain automatic language decision must not hide definitions written for the real language.
        val scopeLang = if (run.language?.uncertain == true && run.language?.source == co.immanuel.arkspeech.core.lang.LanguageSource.AUTO) null else lang
        val active = deps.definitions.activeFor(scopeLang)
        val sw = Stopwatch()
        val classification = if (forcedDefinitionId != null) {
            val def = active.firstOrNull { it.id == forcedDefinitionId } ?: deps.definitions.get(forcedDefinitionId)
            if (def == null) IntentClassification.none()
            else IntentClassification(IntentStatus.USER_SELECTED, IntentCandidate(def.id, def.intent.name, def.version, 1f), engine = "user")
        } else {
            var c = deps.intentClassifier.classify(text, scopeLang, active)
            if (c.status == IntentStatus.NO_MATCH && scopeLang != null) {
                // The language decision may be wrong (TTS English heard as Gujarati, say). Try every enabled
                // definition; anything found is offered to the user rather than accepted silently.
                val wide = deps.intentClassifier.classify(text, null, deps.definitions.activeFor(null))
                if (wide.best != null) {
                    c = wide.copy(status = IntentStatus.LOW_CONFIDENCE)
                    run.warn("Intent '${wide.best.intentName}' is not configured for ${Languages.displayName(scopeLang)}; please confirm")
                }
            }
            c
        }
        run.intent = classification
        run.nluEngine = "${deps.intentClassifier.engineId}+${extractor.engineId}"
        run.stage("intent", sw.elapsedMs(), classification.status.name, classification.best?.let { "${it.definitionId} ${"%.2f".format(it.score)}" })
        log.info(TAG, "intent classified", mapOf("requestId" to run.requestId, "status" to classification.status, "definition" to classification.best?.definitionId, "score" to classification.confidence))

        val best = classification.best
        val def = best?.let { b -> active.firstOrNull { it.id == b.definitionId } ?: deps.definitions.get(b.definitionId) }
        if (def == null || classification.status == IntentStatus.NO_MATCH) {
            run.extraction = null; run.definition = null
            run.status = PipelineStatus.NO_INTENT
            return run
        }
        if (classification.status == IntentStatus.LOW_CONFIDENCE && def.fallback == FallbackBehaviour.REJECT) {
            run.extraction = null; run.definition = def
            run.status = PipelineStatus.NO_INTENT
            run.warn("Intent '${def.intent.name}' scored ${"%.2f".format(classification.confidence)}, below its threshold ${def.confidenceThreshold}; rejected by its fallback rule")
            return run
        }
        val sw2 = Stopwatch()
        val ext = extractor.extract(text, def, lang)
        run.extraction = ext
        run.definition = def
        run.stage("extract", sw2.elapsedMs(), if (ext.needsClarification) "NEEDS_CLARIFICATION" else "OK", "${ext.fields.count { it.value.status == FieldStatus.EXTRACTED }}/${ext.fields.size} extracted")
        log.info(TAG, "fields extracted", mapOf("requestId" to run.requestId, "definition" to def.id, "statuses" to ext.fields.values.groupingBy { it.status }.eachCount()))
        run.status = finalStatus(run)
        return run
    }

    /** The user answered a clarification: validates the answer and updates the field. */
    fun provideFieldValue(run: PipelineRun, field: String, value: String): PipelineRun {
        val def = run.definition ?: return run
        val fd = def.fields.firstOrNull { it.name == field } ?: return run
        val ext = run.extraction ?: return run
        val updated = extractor.userValue(fd, value)
        run.extraction = ext.copy(fields = ext.fields + (field to updated))
        run.stage("clarify", 0, updated.status.name, field)
        run.status = finalStatus(run)
        return run
    }

    /** The user confirmed a LOW_CONFIDENCE value as is. */
    fun confirmFieldValue(run: PipelineRun, field: String): PipelineRun {
        val ext = run.extraction ?: return run
        val f = ext.fields[field] ?: return run
        if (f.value == null) return run
        run.extraction = ext.copy(fields = ext.fields + (field to f.copy(status = FieldStatus.USER_PROVIDED, confidence = 1f)))
        run.status = finalStatus(run)
        return run
    }

    private fun finalStatus(run: PipelineRun): PipelineStatus {
        val ext = run.extraction ?: return PipelineStatus.NO_INTENT
        val intentNeedsUser = run.intent?.status == IntentStatus.AMBIGUOUS || run.intent?.status == IntentStatus.LOW_CONFIDENCE
        val langNeedsUser = run.language?.let { it.uncertain && it.source == co.immanuel.arkspeech.core.lang.LanguageSource.AUTO } == true
        return if (ext.needsClarification || intentNeedsUser || langNeedsUser) PipelineStatus.NEEDS_CLARIFICATION else PipelineStatus.OK
    }

    fun result(run: PipelineRun): PipelineResult = run.toResult(deps.deviceInfo, deps.clock.now())

    /** One-shot: the whole chain with no user interaction; the result says what still needs clarifying. */
    suspend fun runAll(audio: PcmAudio, languageOverride: String? = null, consent: Consent = Consent()): PipelineResult {
        val run = begin(audio, consent)
        if (run.isTerminal) return result(run)
        if (languageOverride != null) overrideLanguage(run, languageOverride) else detectLanguage(run)
        if (run.status == PipelineStatus.UNSUPPORTED_LANGUAGE) return result(run)
        transcribe(run)
        if (run.isTerminal) return result(run)
        extract(run)
        return result(run)
    }

    companion object {
        private const val TAG = "SpeechPipeline"
        /** States in which there is nothing to understand: no usable audio, no model, or an engine failure. */
        private val BLOCKING = setOf(PipelineStatus.AUDIO_REJECTED, PipelineStatus.MODEL_UNAVAILABLE, PipelineStatus.FAILED, PipelineStatus.UNSUPPORTED_LANGUAGE)
    }
}
