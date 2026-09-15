"""
Music generation backend powered by Meta's MusicGen via HuggingFace Transformers.

Two model variants are used depending on whether a melody audio file is supplied:

  • facebook/musicgen-small          – text-to-music only, ~300 MB, fastest on CPU
  • facebook/musicgen-melody         – melody+text conditioned,  ~1.5 GB, slower

For a CPU-only machine the default small/melody-small models are recommended.
On first run each model is downloaded and cached by HuggingFace (~/. cache/huggingface).
"""

from __future__ import annotations

import os
import platform
import sys
import warnings
from typing import Callable, Optional

import numpy as np

import torch

# Silence verbose HF / torch warnings on CPU
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Token rate constants for MusicGen
# ──────────────────────────────────────────────────────────────────────────────
# MusicGen encodes audio at 50 tokens / second (EnCodec frame rate).
_TOKENS_PER_SECOND = 50

# Longest single generation.  MusicGen is trained on 30 s clips and its melody
# conditioning covers exactly 30 s, so 30 s is the natural ceiling; the text
# mode keeps its own tighter 20 s limit in api.py/main.py for CPU time.
MAX_DURATION_SEC   = float(os.environ.get("ARK_MAX_DURATION_SEC", "30"))


def _select_device() -> str:
    """
    Pick the torch device for generation.  Defaults to **CPU**.

    CPU is required on Azure App Service (no GPU) and — perhaps surprisingly — is
    also the fastest option on Apple Silicon for this workload: MusicGen decodes
    autoregressively (one token at a time), so a GPU spends more time on
    per-kernel dispatch/sync than on useful work.  Benchmarked on an M5, the MPS
    path was ~2× *slower* than a multi-threaded CPU decode for the small model.

    Opt into a GPU explicitly to A/B it (worth trying on the larger melody model,
    where each step does more compute so the dispatch overhead matters less):
      ARK_DEVICE=mps      – Apple-Silicon Metal
      ARK_DEVICE=cuda     – NVIDIA
      ARK_FORCE_CPU=1     – always CPU (wins over ARK_DEVICE; set on Azure)
    """
    if os.environ.get("ARK_FORCE_CPU") == "1":
        return "cpu"
    override = os.environ.get("ARK_DEVICE", "").strip().lower()
    if override:
        return override
    return "cpu"


_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def _select_dtype(device: str) -> torch.dtype:
    """
    Weight precision, ``ARK_DTYPE=float32|bfloat16|float16``.

    The melody model is 1.55 B parameters: 6.2 GB in float32 but 3.1 GB in
    bfloat16.  On a box where those 6 GB do not fit next to everything else
    (a 16 GB laptop with a browser and a VM open, or a 3.5 GB Azure B2 plan)
    float32 does not merely run slower — the weights get paged from disk on
    every one of the ~1 500 decode steps and a 30 s clip takes hours.

    Measured on an M5 (10 cores, 16 GB), musicgen-melody, 30 tokens:
        cpu / float32   : swapping, > 7 min for 30 tokens
        cpu / bfloat16  : 0.05 s / token   (30 s clip ≈ 80 s)
        mps / bfloat16  : 0.06 s / token
        mps / float16   : 0.05 s / token
    bfloat16 keeps float32's exponent range, so the T5 text encoder (which
    overflows in float16) is safe.  It is therefore the default on MPS, CUDA
    and Apple-Silicon CPUs.  Other CPUs (Azure's x86 workers) stay on float32
    unless ``ARK_DTYPE`` says otherwise — many of them lack native bf16 matmul.
    """
    name = os.environ.get("ARK_DTYPE", "").strip().lower()
    if name in _DTYPES:
        return _DTYPES[name]
    if device in ("mps", "cuda") or (device == "cpu" and _is_apple_silicon()):
        return torch.bfloat16
    return torch.float32


def _cpu_thread_count() -> int:
    """
    Threads for the CPU decode.  Defaults to every core, override with
    ``ARK_NUM_THREADS``.

    All cores is correct on Azure's homogeneous Linux CPUs.  On Apple Silicon the
    efficiency cores add memory-bandwidth contention, so a value near the
    performance-core count can be a few percent faster — tune with
    ``ARK_NUM_THREADS`` if you care about the last few percent.  The previous
    hard floor of 4 left most of a modern machine idle.
    """
    override = os.environ.get("ARK_NUM_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return os.cpu_count() or 4


def max_new_tokens_for(duration: float) -> int:
    """
    Number of decoder tokens MusicGen will generate for ``duration`` seconds.

    Mirrors exactly what :meth:`MusicGenerator.generate` passes as
    ``max_new_tokens`` so callers (e.g. the job worker's ETA/token-count
    display) can size the progress bar without loading the model.
    """
    duration = float(np.clip(duration, 1.0, MAX_DURATION_SEC))
    return int(duration * _TOKENS_PER_SECOND) + 4  # small headroom


class MusicGenerator:
    """
    Thin wrapper around HuggingFace MusicGen that handles both text-only
    and melody-conditioned generation.

    Parameters
    ----------
    use_melody_model : bool
        When True loads ``facebook/musicgen-melody`` which supports audio
        conditioning.  Falls back to ``facebook/musicgen-small`` automatically
        if the melody variant cannot be loaded.
    model_id : str | None
        Override the model HuggingFace hub path entirely.
    """

    def __init__(
        self,
        use_melody_model: bool = False,
        model_id: str | None = None,
    ) -> None:
        # ── Select model variant ──────────────────────────────────────────────
        if model_id:
            self._model_id = model_id
        elif use_melody_model:
            self._model_id = "facebook/musicgen-melody"
        else:
            self._model_id = "facebook/musicgen-small"

        self._melody_capable = "melody" in self._model_id
        self._model     = None
        self._processor = None
        self._device    = _select_device()
        self._dtype     = _select_dtype(self._device)
        self.sample_rate: int = 32_000   # MusicGen default; updated after load

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoProcessor

        # Any MPS op MusicGen may not implement yet falls back to CPU instead of
        # raising — keeps generation working on Apple Silicon across torch
        # versions.  No-op on the Azure/CPU path.
        if self._device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        print(f"  Loading model  : {self._model_id}")
        print(f"  Device         : {self._device}")
        print(f"  Precision      : {str(self._dtype).replace('torch.', '')}")
        print("  (First run downloads weights – this may take a few minutes)")

        self._processor = AutoProcessor.from_pretrained(self._model_id)

        if self._melody_capable:
            from transformers import MusicgenMelodyForConditionalGeneration
            self._model = MusicgenMelodyForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=self._dtype,
            )
        else:
            from transformers import MusicgenForConditionalGeneration
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=self._dtype,
            )

        self._model.eval()
        self._model.to(self._device)

        # Grab actual sample rate from codec config
        try:
            self.sample_rate = self._model.config.audio_encoder.sampling_rate
        except AttributeError:
            self.sample_rate = 32_000

        # On the CPU path, spread the decode across cores (env-tunable).  On MPS
        # the heavy decode runs on the GPU, so CPU threads stay at torch default.
        if self._device == "cpu":
            torch.set_num_threads(_cpu_thread_count())
        print(f"  Sample rate    : {self.sample_rate} Hz")
        print(f"  Torch threads  : {torch.get_num_threads()}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        melody_path: str | None = None,
        duration: float = 20.0,
        guidance_scale: float = 3.5,
        temperature: float = 1.05,
        top_k: int = 250,
        progress_cb: Optional[Callable[[float], None]] = None,
        continuation: Optional[tuple[np.ndarray, int]] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from a text prompt with optional melody conditioning.

        Parameters
        ----------
        prompt        : MusicGen text prompt
        melody_path   : path to a WAV/MP3/FLAC file whose chromagram
                        conditions the generation  (melody model only)
        duration      : desired output length in seconds (capped at 20 s)
        guidance_scale: classifier-free guidance strength (higher = closer to prompt)
        temperature   : sampling temperature (slightly above 1 adds variety)
        top_k         : nucleus sampling k
        progress_cb   : optional ``callback(fraction: float)`` invoked once per
                        generated audio token with ``fraction`` in [0, 1] — the
                        share of ``max_new_tokens`` produced so far.  Lets a UI
                        track the (otherwise opaque) autoregressive decode loop.
        continuation  : optional ``(audio, sample_rate)`` — a few seconds of
                        music the new clip must *continue from*.  The audio is
                        tokenised with the model's EnCodec and used as the
                        decoder prompt (MusicGen's audio-continuation mode);
                        the returned audio contains only the newly generated
                        ``duration`` seconds, not the prompt.  This is how the
                        vocal pipeline keeps consecutive 30 s windows coherent.

        Returns
        -------
        (audio_numpy, sample_rate)
            audio_numpy has shape (2, samples) for stereo or (samples,) for mono
        """
        self._load()

        duration = float(np.clip(duration, 1.0, MAX_DURATION_SEC))
        max_new_tokens = max_new_tokens_for(duration)

        # ── Build processor inputs ────────────────────────────────────────────
        use_melody = melody_path is not None and self._melody_capable

        if use_melody:
            # Load the melody as a mono float32 numpy array via librosa.  This
            # avoids torchaudio.load, which in torchaudio >= 2.8 delegates to
            # TorchCodec (an extra package + matching FFmpeg) that isn't
            # available on Azure App Service.  librosa already handles the
            # down-mix to mono (mono=True) and preserves the native sample
            # rate (sr=None).
            import librosa

            melody_np, melody_sr = librosa.load(melody_path, sr=None, mono=True)

            # Trim to duration
            max_samples = int(duration * melody_sr)
            melody_np = melody_np[:max_samples]

            inputs = self._processor(
                audio=melody_np,
                sampling_rate=melody_sr,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )

        inputs = {
            k: (v.to(self._device, dtype=self._dtype) if v.is_floating_point()
                else v.to(self._device))
            for k, v in inputs.items()
        }

        # ── Audio continuation (decoder prompt) ───────────────────────────────
        extra_kwargs: dict = {}
        prompt_samples = 0
        if continuation is not None:
            decoder_input_ids, prompt_samples = self._encode_continuation(*continuation)
            if decoder_input_ids is not None:
                extra_kwargs["decoder_input_ids"] = decoder_input_ids

        # ── Generate ──────────────────────────────────────────────────────────
        print(f"  Prompt         : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  Duration       : {duration} s  ({max_new_tokens} tokens)")
        print(f"  Melody guided  : {use_melody}")
        if prompt_samples:
            print(f"  Continuing from: {prompt_samples / self.sample_rate:.1f} s of audio")

        # ── Per-token progress hook ───────────────────────────────────────────
        # MusicGen decodes autoregressively, so transformers invokes any
        # StoppingCriteria once per generated token.  We attach one that never
        # halts generation but reports how far through ``max_new_tokens`` we are.
        stopping = self._build_progress_criteria(max_new_tokens, progress_cb)

        with torch.inference_mode():
            output_tokens = self._model.generate(
                **inputs,
                **extra_kwargs,
                do_sample=True,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping,
            )

        if progress_cb is not None:
            try:
                progress_cb(1.0)   # ensure the bar lands on 100% of this call
            except Exception:
                pass

        # output_tokens shape: (batch, channels, time)
        audio_tensor = output_tokens[0]   # first (only) batch item

        # Convert to float32 numpy, shape (channels, samples) or (samples,)
        audio_np = audio_tensor.cpu().float().numpy()

        # Drop the decoded continuation prompt — callers only want new audio.
        if prompt_samples:
            audio_np = audio_np[..., prompt_samples:]

        # Ensure stereo (duplicate mono if needed)
        if audio_np.ndim == 1:
            audio_np = np.stack([audio_np, audio_np])
        elif audio_np.shape[0] == 1:
            audio_np = np.repeat(audio_np, 2, axis=0)

        # Normalise to [-1, 1]
        peak = np.max(np.abs(audio_np))
        if peak > 0:
            audio_np = audio_np / peak

        return audio_np.astype(np.float32), self.sample_rate

    # ──────────────────────────────────────────────────────────────────────────
    # Audio continuation
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_continuation(
        self, audio: np.ndarray, sr: int,
    ) -> tuple[Optional[torch.Tensor], int]:
        """
        Tokenise ``audio`` with the model's EnCodec so it can prime the decoder.

        Returns ``(decoder_input_ids, prompt_samples)`` where the ids have the
        ``(num_codebooks, frames)`` layout MusicGen's ``generate`` expects and
        ``prompt_samples`` is how much of the decoded output belongs to the
        prompt.  Mirrors what the text-only model does internally for
        ``input_values``; the melody model in transformers has no such hook, so
        it is done here for both.
        """
        encoder = getattr(self._model, "audio_encoder", None)
        if encoder is None:
            return None, 0

        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=0)
        if int(sr) != int(self.sample_rate):
            import librosa
            mono = librosa.resample(mono, orig_sr=int(sr), target_sr=int(self.sample_rate))
        if mono.size < self.sample_rate // 10:          # < 100 ms: not worth it
            return None, 0
        peak = float(np.max(np.abs(mono)))
        if peak > 0:
            mono = mono / peak * 0.9

        hop = int(getattr(self._model.config.audio_encoder, "hop_length", 0) or
                  self.sample_rate // _TOKENS_PER_SECOND)

        x = torch.from_numpy(mono)[None, None, :].to(self._device, dtype=self._dtype)
        with torch.inference_mode():
            codes = encoder.encode(x).audio_codes        # (frames, batch, codebooks, T)
        ids = codes[0, 0]                                # (codebooks, T)
        n_codebooks = int(self._model.decoder.num_codebooks)
        if ids.shape[0] != n_codebooks:
            ids = ids[:n_codebooks]
        prompt_frames = int(ids.shape[-1])
        return ids.reshape(n_codebooks, prompt_frames).to(torch.long), prompt_frames * hop

    # ──────────────────────────────────────────────────────────────────────────
    # Progress hook
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_progress_criteria(
        max_new_tokens: int,
        progress_cb: Optional[Callable[[float], None]],
    ):
        """
        Build a ``StoppingCriteriaList`` that reports decode progress.

        Returns ``None`` when no callback is supplied so ``generate`` runs with
        its default (no) criteria.  The criterion never stops generation — it
        always returns all-False — it only measures how many tokens have been
        produced relative to ``max_new_tokens``.
        """
        if progress_cb is None:
            return None

        from transformers import StoppingCriteria, StoppingCriteriaList

        total = max(int(max_new_tokens), 1)

        class _ProgressCriteria(StoppingCriteria):
            def __init__(self) -> None:
                self._start_len: int | None = None

            def __call__(self, input_ids, scores, **kwargs):
                cur = input_ids.shape[-1]
                # First call establishes the prompt/BOS baseline length.
                if self._start_len is None:
                    self._start_len = cur
                step = cur - self._start_len
                frac = min(max(step / total, 0.0), 1.0)
                try:
                    progress_cb(frac)
                except Exception:
                    pass
                # Never request a stop — return one False per sequence in batch.
                return torch.zeros(
                    input_ids.shape[0], dtype=torch.bool, device=input_ids.device
                )

        return StoppingCriteriaList([_ProgressCriteria()])
