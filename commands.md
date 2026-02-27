# AudioCapBench - Evaluation Commands

All commands assume you have already:
1. Activated the venv: `source .venv/bin/activate`
2. Built the dataset: `python -m audiocapbench.build_dataset --output-dir data/audio_caption`
3. Authenticated GCP: `source ~/google-cloud-sdk/path.bash.inc && gcloud auth application-default login --project salesforce-research-internal`

---

## OpenAI Models (via Salesforce Gateway)

### gpt-4o-audio-preview
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gpt-4o-mini-audio-preview
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-mini-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gpt-audio
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-audio \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gpt-audio-mini
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-audio-mini \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

---

## OpenAI Realtime Models (via Salesforce Gateway WebSocket)

Note: Realtime models use WebSocket connections. Concurrency is limited since each connection is persistent. Use `--concurrency 3` to avoid overwhelming the gateway.

### gpt-4o-realtime-preview
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai_realtime --model gpt-4o-realtime-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 3 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gpt-realtime
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai_realtime --model gpt-realtime \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 3 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gpt-realtime-mini
```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai_realtime --model gpt-realtime-mini \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 3 \
    --max-tokens 8192 \
    --no-aac-metrics
```

---

## Gemini Models (via GCP Vertex AI)

### gemini-2.5-flash
```bash
source credentials.env && source ~/google-cloud-sdk/path.bash.inc && python -m audiocapbench.evaluate \
    --provider gemini --model gemini-2.5-flash \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gemini-2.5-pro
```bash
source credentials.env && source ~/google-cloud-sdk/path.bash.inc && python -m audiocapbench.evaluate \
    --provider gemini --model gemini-2.5-pro \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gemini-2.5-flash-lite
```bash
source credentials.env && source ~/google-cloud-sdk/path.bash.inc && python -m audiocapbench.evaluate \
    --provider gemini --model gemini-2.5-flash-lite \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gemini-2.0-flash
```bash
source credentials.env && source ~/google-cloud-sdk/path.bash.inc && python -m audiocapbench.evaluate \
    --provider gemini --model gemini-2.0-flash \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gemini-3-flash-preview
Note: Gemini 3 requires `VERTEX_LOCATION=global` (not us-central1).
```bash
source credentials.env && source ~/google-cloud-sdk/path.bash.inc && VERTEX_LOCATION=global python -m audiocapbench.evaluate \
    --provider gemini --model gemini-3-flash-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 \
    --max-tokens 8192 \
    --no-aac-metrics
```

### gemini-3-pro-preview
```bash
source credentials.env && source ~/google-cloud-sdk/path.bash.inc && VERTEX_LOCATION=global python -m audiocapbench.evaluate \
    --provider gemini --model gemini-3-pro-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 20 \
    --max-tokens 8192 \
    --no-aac-metrics
```

---

## Quick Test (10 samples, no LLM judge)

```bash
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --max-samples 10 \
    --concurrency 5 \
    --no-aac-metrics --no-llm-judge
```

## Single Category Test

```bash
# Sound only
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --category sound \
    --concurrency 10 \
    --no-aac-metrics

# Music only
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --category music \
    --concurrency 10 \
    --no-aac-metrics

# Speech only
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --category speech \
    --concurrency 10 \
    --no-aac-metrics
```

---

## Notes

- **Concurrency**: `--concurrency 10` runs 10 parallel API calls for both inference and LLM judge. Adjust based on rate limits.
- **max-tokens**: `--max-tokens 8192` is used uniformly across all models for fairness. Gemini 2.5+ thinking models require it (thinking tokens share the budget), and GPT models benefit from not being artificially truncated at the default 256.
- **Results**: Saved to `results/<provider>_<model>_<timestamp>.json`
- **LLM Judge**: Uses GPT-4.1 via Salesforce Gateway. Disable with `--no-llm-judge` for faster runs.
- **Some models may not be available** on the Salesforce Gateway or Vertex AI. If a model returns errors, it may not be enabled for your project.
