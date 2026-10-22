---
title: EgoRA Rotation Diagnostics
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Measure fine-tuning rotation; EgoRA vs plain LoRA live
---

# EgoRA — Rotation Diagnostics

Interactive dashboard for [EgoRA](https://github.com/ArsSocratica/EgoRA):

- **Diagnose** — rotation geometry between a base model and a fine-tuned model or LoRA adapter,
  per projection or per attention head, with a model-specific θ_crit = arcsin(1/√d_head).
- **Model Info** — head dimension, θ_crit and LoRA parameter counts (config only).
- **EgoRA vs LoRA** — short side-by-side fine-tune tracking rotation, loss and forgetting.

## Deploy

Copy this folder into a new Gradio Space (or push it as the Space repo root). On the free CPU
tier, keep to models up to ~0.5B parameters. Set `EGORA_ALLOW_TRAINING=0` to hide the
comparison tab.
