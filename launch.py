"""ACE-Step 1.5 — Gradio UI launcher for DGX Spark."""

import os
import sys

# Bind to all interfaces for Docker port-mapping
os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")

# Patch torchaudio.save — torchaudio 2.10 routes all saves through torchcodec
# which has an ABI mismatch with torch 2.10+cu130 on aarch64. We monkey-patch
# torchaudio.save to use soundfile directly.  The patch MUST live in this
# process (not a child via execvp) so that every later `import torchaudio`
# picks up the patched function.
import torch
import soundfile as sf
import torchaudio

_original_save = torchaudio.save          # keep a reference just in case

def _patched_save(filepath, src, sample_rate, channels_first=True, format=None, **kwargs):
    """Save audio using soundfile, bypassing broken torchcodec in torchaudio 2.10."""
    if isinstance(src, torch.Tensor):
        src = src.cpu().float()
        if channels_first and src.dim() == 2:
            src = src.T  # [channels, samples] -> [samples, channels]
        src = src.numpy()
    # Infer format from extension if not given
    if format is None:
        ext = str(filepath).rsplit(".", 1)[-1].lower()
        fmt_map = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}
        format = fmt_map.get(ext, ext.upper())
    else:
        format = format.upper()
    subtype = "FLOAT" if format == "WAV" and kwargs.get("bits_per_sample") == 32 else None
    sf.write(str(filepath), src, sample_rate, format=format, subtype=subtype)

torchaudio.save = _patched_save
print("[ACE-Step] Patched torchaudio.save → soundfile backend", flush=True)

# Build sys.argv as if we had launched:
#   python -m acestep.acestep_v15_pipeline --server-name 0.0.0.0 ...
argv = [
    "acestep.acestep_v15_pipeline",
    "--server-name", "0.0.0.0",
    "--port", os.environ.get("PORT", "7860"),
    "--backend", os.environ.get("ACESTEP_LLM_BACKEND", "pt"),
]

# Auto-init models on startup
if os.environ.get("ACESTEP_INIT_SERVICE", "true").lower() == "true":
    config_path = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
    lm_model_path = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-4B")
    argv.extend([
        "--init_service", "true",
        "--config_path", config_path,
        "--init_llm", "true",
        "--lm_model_path", lm_model_path,
    ])

sys.argv = argv
print(f"[ACE-Step] Launching in-process: {' '.join(argv)}", flush=True)

# Run the module in THIS process so the torchaudio patch stays active.
import runpy
runpy.run_module("acestep.acestep_v15_pipeline", run_name="__main__")
