# newdream-sdxl

An experiment in speeding up SDXL inference by blending a short standard denoising run with a one-step LCM-LoRA run, plus a script that measures the speed and similarity of the result against a 20-step baseline.

## Overview

The target model is [`stablediffusionapi/newdream-sdxl-20`](https://huggingface.co/stablediffusionapi/newdream-sdxl-20), an SDXL checkpoint. The goal is to produce outputs that stay close to the model's normal 20-step output in less time.

The repository contains one script, `benchmark.py`, with two parts.

**Baseline generation (`generate`)**
The unmodified pipeline in float16 runs 20 inference steps and returns latents.

**Candidate generation (`efficient_generate`)**
For the same prompt and seed:

1. The standard pipeline runs 10 inference steps.
2. A second copy of the pipeline, with an `LCMScheduler` and the LCM-LoRA adapter [`mhussainahmad/sdxl-lcmlora-1024-100k-3000steps`](https://huggingface.co/mhussainahmad/sdxl-lcmlora-1024-100k-3000steps) loaded, runs 1 step with `guidance_scale=0.01`.
3. The two sets of latents are blended as `0.8 * standard + 0.2 * lcm_lora`.

Timing covers both passes.

**Evaluation (`compare_checkpoints`)**

- 5 random prompts are built from adjectives and nouns in the NLTK `words` corpus, filtered with a part-of-speech tagger. Each prompt gets a random seed.
- Baseline latents and timings are recorded first. The candidate method is then run on the same prompt and seed pairs.
- Similarity is `(cosine_similarity(baseline, candidate) * 0.5 + 0.5) ** 4`, computed on flattened latents.
- The score is `max(0, BASELINE_AVERAGE - average_time) * average_similarity`. `BASELINE_AVERAGE` is a fixed constant in the script (2.58), not a measured result.
- The loop stops early if the projected speed cannot beat the measured baseline, or if average similarity falls below 0.85.

The evaluation logic appears to be adapted from the WOMBO edge-maxxing SDXL contest benchmark. This repository contains no saved benchmark output, so no measured speedup or similarity figures are reported here.

## Repository layout

```
benchmark.py       Baseline vs. blended LCM-LoRA generation, timing and similarity scoring
requirements.txt   Pinned dependencies
```

## Getting started

You need a CUDA GPU with enough memory to hold two SDXL pipelines in float16. The models and the LoRA adapter are downloaded from Hugging Face on first run, and the NLTK corpora are downloaded at import time.

```bash
pip install -r requirements.txt
python benchmark.py
```

The script prints per-sample generation time and similarity, followed by the average similarity, average time, and final score.

## Tech stack

Python, PyTorch 2.2, Hugging Face Diffusers 0.29 (`StableDiffusionXLPipeline`, `LCMScheduler`), Transformers, Accelerate, PEFT (LoRA loading), NLTK.

## Credits

- Base model: [`stablediffusionapi/newdream-sdxl-20`](https://huggingface.co/stablediffusionapi/newdream-sdxl-20), derived from Stability AI's SDXL. See the model card for license terms.
- Latent Consistency Model LoRA method: Luo et al., "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module" (2023).
