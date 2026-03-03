# AudioCapBench

A benchmark for evaluating audio captioning models across three domains: **environmental sound**, **music**, and **speech**.

1,000 samples | LLM-as-Judge + reference metrics

## Quick Setup

```bash
# 1. Install dependencies
bash install.sh
source .venv/bin/activate

# 2. Set up credentials
cp credentials.env.template credentials.env
# Edit credentials.env with your API keys (OpenAI, Gemini, HuggingFace)

# 3. Build evaluation dataset (downloads audio from HuggingFace)
source credentials.env
python -m audiocapbench.build_dataset --output-dir data/audio_caption
```

## Quick Evaluation

```bash
# Evaluate a model
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --concurrency 10 --max-tokens 8192 --no-aac-metrics

# Quick test (10 samples, no LLM judge)
source credentials.env && python -m audiocapbench.evaluate \
    --provider openai --model gpt-4o-audio-preview \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --max-samples 10 --no-aac-metrics

# Single category
source credentials.env && python -m audiocapbench.evaluate \
    --provider gemini --model gemini-2.5-flash \
    --data-dir data/audio_caption \
    --credentials credentials.env \
    --category music --concurrency 10 --max-tokens 8192 --no-aac-metrics
```
---
# More Details

## Supported Models

| Provider | Models | API Type |
|----------|--------|----------|
| OpenAI | `gpt-4o-audio-preview`, `gpt-audio`, `gpt-audio-mini`, `gpt-4o-mini-audio-preview` | Chat Completions |
| OpenAI | `gpt-4o-realtime-preview`, `gpt-realtime`, `gpt-realtime-mini` | Realtime WebSocket |
| Gemini | `gemini-2.0-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`, `gemini-3-pro-preview` | Vertex AI / API key |

## Evaluation Dataset

| Category | Source | Samples |
|----------|--------|---------|
| Sound | [Clotho v2](https://huggingface.co/datasets/piyushsinghpasi/clotho-multilingual) test + [AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) test | 200 + 200 |
| Music | [MusicCaps](https://huggingface.co/datasets/kelvincai/MusicCaps_30s_wav) eval set | 300 |
| Speech | [Emo Speech Caption](https://huggingface.co/datasets/seastar105/emo_speech_caption_test) | 300 |
| **Total** | | **1,000** |

## Evaluation Metrics

**LLM-as-Judge** (GPT-4.1): Accuracy, Completeness, Hallucination (each 0-10). Overall = average of all three.

**Reference-based**: METEOR, BLEU-4, ROUGE-L (via NLTK + rouge-score).

## Credentials

Copy the template and fill in your keys:

```bash
cp credentials.env.template credentials.env
```

| Variable | Required for | How to get |
|----------|-------------|------------|
| `OPENAI_API_KEY` | OpenAI models + LLM judge | [platform.openai.com](https://platform.openai.com) |
| `GEMINI_API_KEY` | Gemini models (API key mode) | [aistudio.google.com](https://aistudio.google.com) |
| `VERTEX_PROJECT` | Gemini models (Vertex AI mode) | GCP project ID + `gcloud auth` |
| `HF_TOKEN` | Dataset download | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Project Structure

```
AudioCapBench/
├── audiocapbench/           # Main package
│   ├── build_dataset.py     # Dataset builder (downloads from HuggingFace)
│   ├── evaluate.py          # Evaluation pipeline
│   ├── models.py            # Model clients (OpenAI, Gemini, Qwen)
│   ├── metrics.py           # Metrics (aac-metrics, NLTK fallback, LLM judge)
│   └── config.py            # Config & credential loading
├── eval_data_ids/           # Curated 1000-sample eval subset (CSV)
├── configs/default.yaml     # Default configuration
├── install.sh               # Setup script
├── credentials.env.template # Credentials template
└── results/                 # Evaluation results (JSON)
```

## License

This project is licensed under Apache-2.0 license.

Individual datasets retain their original licenses:
- Clotho: [Tampere University License](https://zenodo.org/record/3490684)
- AudioCaps: CC-BY-NC-4.0
- MusicCaps: CC-BY-SA-4.0
- Emo Speech Caption: See [dataset card](https://huggingface.co/datasets/seastar105/emo_speech_caption_test)

## Citation

If you find our project useful, here is our paper:

```
@article{Qiu2025LoCoBenchAgentAI,
  title={AudioCapBench: Quick Evaluation on Audio Captioning across
Sound, Music, and Speech},
  author={Jielin Qiu, Jianguo Zhang, Zixiang Chen, Liangwei Yang, Ming Zhu, Juntao Tan, Haolin Chen, Wenting Zhao, Rithesh Murthy, Roshan Ram, Akshara Prabhakar, Shelby Heinecke, Caiming, Xiong, Silvio Savarese, Huan Wang},
  journal={ArXiv},
  year={2026},
  volume={abs/2602.23649}
}
```


