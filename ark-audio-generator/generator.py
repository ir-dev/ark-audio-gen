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

# Cap output to 20 s on CPU to avoid very long runtimes
MAX_DURATION_SEC   = 20.0


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
        print("  (First run downloads weights – this may take a few minutes)")

        self._processor = AutoProcessor.from_pretrained(self._model_id)

        if self._melody_capable:
            from transformers import MusicgenMelodyForConditionalGeneration
            self._model = MusicgenMelodyForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=torch.float32,
            )
        else:
            from transformers import MusicgenForConditionalGeneration
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self._model_id,
                torch_dtype=torch.float32,
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

        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # ── Generate ──────────────────────────────────────────────────────────
        print(f"  Prompt         : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  Duration       : {duration} s  ({max_new_tokens} tokens)")
        print(f"  Melody guided  : {use_melody}")

        # ── Per-token progress hook ───────────────────────────────────────────
        # MusicGen decodes autoregressively, so transformers invokes any
        # StoppingCriteria once per generated token.  We attach one that never
        # halts generation but reports how far through ``max_new_tokens`` we are.
        stopping = self._build_progress_criteria(max_new_tokens, progress_cb)

        with torch.no_grad():
            output_tokens = self._model.generate(
                **inputs,
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
