# AudioCapBench - Supported Models

## Overview

AudioCapBench evaluates API-based models that accept audio input and generate text captions. Models are organized by provider and API type.

---

## OpenAI

All OpenAI models are accessed via the Salesforce Research Gateway.

### Chat Completions API (audio input -> text output)

| Model ID | Description | Status |
|----------|-------------|--------|
| `gpt-4o-audio-preview` | GPT-4o with audio input/output | Tested, working |
| `gpt-4o-mini-audio-preview` | Smaller, cheaper variant | Available |
| `gpt-audio` | Latest GPT Audio model | Available |
| `gpt-audio-1.5` | Best voice model for audio in Chat Completions | Available |
| `gpt-audio-mini` | Cost-efficient GPT Audio | Available |

### Realtime API (WebSocket, audio input -> audio/text output)

| Model ID | Description | Status |
|----------|-------------|--------|
| `gpt-realtime` | Realtime audio I/O | Requires WebSocket client |
| `gpt-realtime-1.5` | Best realtime voice model | Requires WebSocket client |
| `gpt-realtime-mini` | Cost-efficient realtime | Requires WebSocket client |
| `gpt-4o-realtime-preview` | GPT-4o realtime | Requires WebSocket client |
| `gpt-4o-mini-realtime-preview` | GPT-4o mini realtime | Requires WebSocket client |

Note: Realtime models use the WebSocket-based Realtime API, not the standard Chat Completions API. A separate client implementation is needed.

### Transcription Models (audio -> transcript only)

| Model ID | Description | Status |
|----------|-------------|--------|
| `gpt-4o-transcribe` | GPT-4o powered transcription | Transcription only |
| `gpt-4o-mini-transcribe` | GPT-4o mini transcription | Transcription only |
| `whisper-1` | General-purpose ASR | Transcription only |

Note: Transcription models produce verbatim speech-to-text only, not descriptive captions. They can serve as a baseline for the speech category.

---

## Google Gemini

All Gemini models are accessed via GCP Vertex AI. Authentication via `gcloud auth application-default login`.

### Gemini 3.x (Latest Generation)

| Model ID | Description | Audio Support | Status |
|----------|-------------|---------------|--------|
| `gemini-3.1-pro-preview` | Advanced intelligence, agentic capabilities | Yes | Available |
| `gemini-3-pro-preview` | State-of-the-art multimodal reasoning | Yes | Available |
| `gemini-3-flash-preview` | Frontier-class at lower cost | Yes | Available |

### Gemini 2.5 (Current Stable)

| Model ID | Description | Audio Support | Status |
|----------|-------------|---------------|--------|
| `gemini-2.5-flash` | Fast, good price-performance, thinking model | Yes | Tested, working |
| `gemini-2.5-pro` | Most capable 2.5 model | Yes | Available |
| `gemini-2.5-flash-lite` | Fastest, most budget-friendly | Yes | Available |

### Gemini 2.0 (Previous Generation)

| Model ID | Description | Audio Support | Status |
|----------|-------------|---------------|--------|
| `gemini-2.0-flash` | Previous gen flash model | Yes | Available |

### Audio Specifications

- Maximum audio length: ~8.4 hours (up to 1M tokens)
- Supported formats: MP3, WAV, FLAC, OGG, AAC
- Gemini 2.5 models are "thinking" models: set `thinking_budget` and high `max_output_tokens` to avoid truncation
- Gemini tends to output markdown formatting; the client strips `**bold**` and `*italic*` for cleaner evaluation

---

## ElevenLabs

ElevenLabs offers a Speech-to-Text API with audio event detection capabilities.

### Speech-to-Text API

| Model ID | Description | Status |
|----------|-------------|--------|
| `scribe_v1` | STT with audio event tagging | Requires implementation |
| `scribe_v2` | Improved STT with diarization, multi-channel | Requires implementation |

**Endpoint:** `POST https://api.elevenlabs.io/v1/speech-to-text`

**Capabilities:**
- Transcription with word-level timestamps
- Audio event tagging: detects `(laughter)`, `(footsteps)`, `(music)`, etc.
- Speaker diarization (up to 32 speakers)
- Multi-channel processing (up to 5 channels)

**Limitations for captioning:**
- Primarily a transcription model, not a general audio captioning model
- Audio event tags are supplementary annotations, not full descriptions
- Best suited for speech category; limited for music/environmental sound description
- Can serve as a baseline: transcript + audio events = basic caption

**Authentication:** Requires `ELEVENLABS_API_KEY` environment variable.

---

## Models NOT Supported (No Audio Input API)

| Provider | Reason |
|----------|--------|
| Claude (Anthropic) | Text + image input only. No audio support. |
| Grok (xAI) | Text + image input only. No audio support. |
| Mistral | Text + image input only. No audio support. |
| Llama (Meta) | Text only via API. No audio support. |
| Cohere | Text only. No audio support. |

---

## Running Evaluations

### OpenAI (Chat Completions)
```bash
# Default model
python -m audiocapbench.evaluate --provider openai --data-dir data/audio_caption --credentials credentials.env

# Specific model
python -m audiocapbench.evaluate --provider openai --model gpt-audio-1.5 --data-dir data/audio_caption --credentials credentials.env
```

### Gemini (Vertex AI)
```bash
# Requires: gcloud auth application-default login --project salesforce-research-internal

# Default model (gemini-2.5-flash)
python -m audiocapbench.evaluate --provider gemini --data-dir data/audio_caption --credentials credentials.env --max-tokens 8192

# Gemini 3
python -m audiocapbench.evaluate --provider gemini --model gemini-3-flash-preview --data-dir data/audio_caption --credentials credentials.env --max-tokens 8192
```

### Quick Test (10 samples, no LLM judge)
```bash
python -m audiocapbench.evaluate --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption --credentials credentials.env \
    --max-samples 10 --no-aac-metrics --no-llm-judge
```

---

## Credentials Setup

```bash
# credentials.env

# OpenAI via Salesforce Research Gateway
export OPENAI_API_KEY="your-gateway-key"
export OPENAI_BASE_URL="https://gateway.salesforceresearch.ai/openai/process/v1"

# Gemini via GCP Vertex AI (uses gcloud auth, no API key needed)
export VERTEX_PROJECT="salesforce-research-internal"
export VERTEX_LOCATION="us-central1"

# ElevenLabs (if using scribe models)
export ELEVENLABS_API_KEY="your-elevenlabs-key"

# HuggingFace (for dataset downloads)
export HF_TOKEN="your-hf-token"
```
