# ACE-Step 1.5 — Spark AI Hub

Thin container wrapper for running [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) on NVIDIA DGX Spark.

- **Base image:** `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04`
- **Platform:** `linux/arm64` (DGX Spark GB10)
- **UI:** Gradio on port 7860

## Quick start

```bash
docker run --gpus all -p 7860:7860 \
  -v acestep-checkpoints:/app/checkpoints \
  -v acestep-hf-cache:/root/.cache/huggingface \
  ghcr.io/hitechcloudvn/acestep-spark-ai-hub:latest
```

Models are downloaded on first run and cached in the volumes.
