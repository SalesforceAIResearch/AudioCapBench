# AudioCapBench

Audio Captioning Benchmark for evaluating model performance on music, sound, and speech audio captioning.

## Project Structure

```
AudioCapBench/
├── audiocapbench/              # Main package
│   ├── __init__.py
│   ├── build_dataset.py        # Dataset builder (downloads audio from HuggingFace)
│   ├── config.py               # Credentials & config loading
│   ├── evaluate.py             # Evaluation pipeline
│   ├── metrics.py              # aac-metrics + fallback NLTK metrics + LLM judge
│   └── models.py               # API model clients (OpenAI, Gemini, Qwen-Audio)
├── configs/
│   └── default.yaml            # Default configuration
├── data_ids/                   # Full sample pools (all available samples per dataset)
│   ├── clotho_ids.csv          # 1045 Clotho v2 test samples
│   ├── audiocaps_ids.csv       # 883 AudioCaps test samples
│   ├── musiccaps_ids.csv       # 903 MusicCaps eval-set samples
│   └── speech_ids.csv          # 5000 speech emotion caption samples
├── eval_data_ids/              # Curated evaluation subset (1000 samples)
│   ├── clotho_eval.csv         # 200 sound samples
│   ├── audiocaps_eval.csv      # 200 sound samples
│   ├── musiccaps_eval.csv      # 300 music samples
│   └── speech_eval.csv         # 300 speech samples
├── credentials.env             # API keys (OPENAI, HF_TOKEN, GOOGLE, etc.)
├── requirements.txt            # Python dependencies
└── reference_code/             # Original reference code (unchanged)
```

## Data Sources

All data is downloaded from HuggingFace. No local data dependencies.

| Category | HuggingFace Repo | Split | Total Available | Eval Subset |
|----------|-----------------|-------|-----------------|-------------|
| Sound (Clotho) | `piyushsinghpasi/clotho-multilingual` | test | 1,045 clips | 200 |
| Sound (AudioCaps) | `OpenSound/AudioCaps` | test | 883 clips | 200 |
| Music (MusicCaps) | `kelvincai/MusicCaps_30s_wav` + `google/MusicCaps` | eval set (`is_audioset_eval=True`) | 903 clips | 300 |
| Speech | `seastar105/emo_speech_caption_test` | train (entire dataset is test) | 5,000 clips | 300 |

## Evaluation Subset

The evaluation subset (`eval_data_ids/`) contains 1,000 curated samples selected for balanced caption lengths.

| File | Samples | Caption Length Range | Avg Caption Length | Selection Criteria |
|------|---------|---------------------|-------------------|-------------------|
| `clotho_eval.csv` | 200 | 77-119 chars | 90 chars | Longest 200 by caption length |
| `audiocaps_eval.csv` | 200 | 78-189 chars | 104 chars | Longest 200 by caption length |
| `musiccaps_eval.csv` | 300 | 86-350 chars | 264 chars | Random sample from 86-350 char range (seed=42) |
| `speech_eval.csv` | 300 | 140-230 chars | 182 chars | Random sample from 140-230 char range (seed=42) |
| **Total** | **1,000** | | | |

## Supported Models

Models with native audio input support:

| Provider | Model ID | Audio Format | Python SDK |
|----------|----------|-------------|------------|
| OpenAI | `gpt-4o-audio-preview` | Base64 WAV/MP3 | `openai` |
| Google Gemini | `gemini-2.5-flash` | Inline bytes | `google-genai` |
| Alibaba Qwen-Audio | `qwen-audio-turbo` | Base64 via DashScope | `dashscope` |

Note: Claude, ElevenLabs, and Grok do not support audio input via their APIs.

## Evaluation Metrics

### LLM-as-Judge (Primary)

The LLM judge (GPT-4.1 via Salesforce Gateway) scores each prediction on 3 orthogonal dimensions:

| Dimension | Scale | What it measures | Analogy |
|-----------|-------|-----------------|---------|
| **Accuracy** | 0-10 | Are the described sounds/events semantically correct? | Precision |
| **Completeness** | 0-10 | Are all key elements from the references covered? | Recall |
| **Hallucination** | 0-10 | Does the prediction avoid inventing content not in the references? (10=none, 0=heavy) | 1 - False Positive Rate |
| **Overall** | 0-10 | Simple average of the 3 dimensions | |

These 3 dimensions were chosen for orthogonality — each captures an independent failure mode:

| | Actually in audio | NOT in audio |
|--|---|---|
| **Model describes it** | Accuracy (correct) | Hallucination (fabricated) |
| **Model doesn't describe it** | Completeness (missed) | — |

**Design decisions and rejected alternatives:**
- **Fluency** was removed because modern LLMs consistently score 9+/10 — it doesn't discriminate between models.
- **Specificity** was removed because it overlaps with Accuracy — a prediction that correctly identifies "a German Shepherd barking" vs "a dog" would already score higher on Accuracy when the reference mentions the breed.
- **Temporal Awareness** was removed because only ~50% of reference captions contain temporal structure ("first X, then Y"). For samples without temporal references, the score defaults to 5/10 with no signal.
- **Descriptive Quality** was considered but overlaps with Completeness (detailed references → matching details = both high) and creates tension with Hallucination (adding detail risks inventing content).

The judge uses category-specific guidance:
- **Sound**: evaluates identification of sound sources, events, acoustic environment
- **Music**: evaluates genre, instrumentation, tempo, mood/atmosphere
- **Speech**: evaluates speaker characteristics, emotional tone, speaking style, and transcript content

### Reference-Based Metrics (Secondary)

Traditional captioning metrics computed against ground-truth reference captions:

**Fallback metrics (always available):**
- **METEOR** - Unigram matching with synonyms and stemming (max across references)
- **BLEU-1/2/3/4** - N-gram precision with smoothing
- **ROUGE-L** - Longest common subsequence F1

**aac-metrics (optional, requires Java 1.8+):**
- **CIDEr-D** - TF-IDF weighted n-gram similarity (standard in image/audio captioning)
- **SPICE** - Semantic propositional content via scene graphs
- **SPIDEr** - (CIDEr-D + SPICE) / 2, standard DCASE ranking metric
- **FENSE** - Fluency-enhanced sentence-BERT similarity (DCASE 2024 ranking metric)

**Why both LLM judge and reference-based metrics?**
- Reference-based metrics (BLEU, METEOR, etc.) penalize correct predictions that use different wording. A model that says "fireworks exploding" when the reference says "multiple explosions going off" gets a low BLEU score despite being semantically correct.
- The LLM judge evaluates semantic similarity and handles paraphrasing. However, it introduces judge model bias and costs API calls.
- Using both provides complementary signal: reference metrics are deterministic and free; LLM judge handles semantic equivalence.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Build the evaluation dataset (downloads audio from HuggingFace)
python -m audiocapbench.build_dataset --output-dir data/audio_caption

# Evaluate with a specific model
python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption

# Evaluate with all models
python -m audiocapbench.evaluate --all-models --data-dir data/audio_caption

# Evaluate a specific category only
python -m audiocapbench.evaluate --provider gemini --category music
```

## CSV File Formats

### data_ids/clotho_ids.csv
| Column | Description |
|--------|-------------|
| `audio_name` | Unique audio clip identifier |
| `duration_s` | Audio duration in seconds |
| `num_captions` | Number of reference captions (typically 5) |
| `sample_caption` | First reference caption |

### data_ids/audiocaps_ids.csv
| Column | Description |
|--------|-------------|
| `youtube_id` | YouTube video identifier |
| `duration_s` | Audio duration in seconds |
| `num_captions` | Number of reference captions (typically 5) |
| `sample_caption` | First reference caption |

### data_ids/musiccaps_ids.csv
| Column | Description |
|--------|-------------|
| `ytid` | YouTube video identifier |
| `start_s` | Clip start time in the video |
| `end_s` | Clip end time in the video |
| `caption` | Free-text music description by professional musician |
| `aspect_list` | Comma-separated musical attributes |

### data_ids/speech_ids.csv
| Column | Description |
|--------|-------------|
| `index` | Sequential index (sorted by caption length desc) |
| `transcript` | What the speaker says (used as unique ID for matching) |
| `caption` | Emotional/paralinguistic description of the speaker |
| `duration_s` | Audio duration in seconds |
