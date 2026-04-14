"""
Music generation backend powered by Meta's MusicGen via HuggingFace Transformers.

Two model variants are used depending on whether a melody audio file is supplied:

  • facebook/musicgen-small          – text-to-music only, ~300 MB, fastest on CPU
  • facebook/musicgen-melody         – melody+text conditioned,  ~1.5 GB, slower

For a CPU-only machine the default small/melody-small models are recommended.
On first run each model is downloaded and cached by HuggingFace (~/. cache/huggingface).
"""

from __future__ import annotations

import warnings
import numpy as np

import torch
import torchaudio

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
        self.sample_rate: int = 32_000   # MusicGen default; updated after load

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoProcessor

        print(f"  Loading model  : {self._model_id}")
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
        self._model.to("cpu")

        # Grab actual sample rate from codec config
        try:
            self.sample_rate = self._model.config.audio_encoder.sampling_rate
        except AttributeError:
            self.sample_rate = 32_000

        # Maximise CPU throughput
        torch.set_num_threads(max(torch.get_num_threads(), 4))
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

        Returns
        -------
        (audio_numpy, sample_rate)
            audio_numpy has shape (2, samples) for stereo or (samples,) for mono
        """
        self._load()

        duration = float(np.clip(duration, 1.0, MAX_DURATION_SEC))
        max_new_tokens = int(duration * _TOKENS_PER_SECOND) + 4  # small headroom

        # ── Build processor inputs ────────────────────────────────────────────
        use_melody = melody_path is not None and self._melody_capable

        if use_melody:
            melody_waveform, melody_sr = torchaudio.load(melody_path)
            # Down-mix to mono for chroma conditioning
            if melody_waveform.shape[0] > 1:
                melody_waveform = melody_waveform.mean(dim=0, keepdim=True)

            # Trim to duration
            max_samples = int(duration * melody_sr)
            melody_waveform = melody_waveform[:, :max_samples]

            inputs = self._processor(
                audio=melody_waveform.squeeze(0).numpy(),
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

        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        # ── Generate ──────────────────────────────────────────────────────────
        print(f"  Prompt         : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"  Duration       : {duration} s  ({max_new_tokens} tokens)")
        print(f"  Melody guided  : {use_melody}")

        with torch.no_grad():
            output_tokens = self._model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )

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
