#!/usr/bin/env python3
"""
Audio Caption Evaluation Pipeline

Evaluates audio captioning models on the AudioCapBench test set.
Supports multiple API-based models and comprehensive metrics.

Usage:
    # Evaluate with a specific model
    python -m audiocapbench.evaluate \
        --provider openai --model gpt-4o-audio-preview \
        --data-dir data/audio_caption \
        --output results/openai_results.json

    # Evaluate with Gemini
    python -m audiocapbench.evaluate \
        --provider gemini --model gemini-2.5-flash \
        --data-dir data/audio_caption

    # Evaluate with all configured models
    python -m audiocapbench.evaluate \
        --all-models --data-dir data/audio_caption

    # Use credentials file
    python -m audiocapbench.evaluate \
        --provider openai --credentials credentials.env \
        --data-dir data/audio_caption
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_credentials
from .metrics import (
    _check_aac_metrics,
    compute_aac_metrics,
    compute_fallback_metrics_batch,
    evaluate_with_llm_judge,
    setup_llm_judge,
)
from .models import AudioCaptionModel, create_model, list_providers

# Prompt instructions per audio category
CATEGORY_INSTRUCTIONS = {
    "sound": [
        "Describe what you hear in this audio.",
        "What sounds can you identify in this audio clip?",
    ],
    "music": [
        "Describe this music clip in detail, including genre, instrumentation, tempo, and mood.",
        "Characterize this musical excerpt with rich detail; cover genre, instrumentation, and overall atmosphere.",
    ],
    "speech": [
        "Describe the speaker and what they are saying, including their tone, emotion, and speaking style.",
        "Describe this speech audio, including the speaker's characteristics and what is being said.",
    ],
}

# URL sanitization
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MD_LINK_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\([^)]+\)")


def sanitize_output(text: str) -> str:
    """Remove URL artifacts and clean model output."""
    if not text:
        return text
    cleaned = MD_LINK_PATTERN.sub("", text)
    cleaned = URL_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _infer_single(
    model: AudioCaptionModel,
    sample: dict,
    index: int,
    total: int,
    data_dir: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> dict:
    """Run inference on a single sample with retries. Thread-safe."""
    sample_id = sample["id"]
    category = sample["category"]
    audio_file = os.path.join(data_dir, sample["audio_file"])
    references = sample["reference_captions"]

    instructions = CATEGORY_INSTRUCTIONS.get(
        category, CATEGORY_INSTRUCTIONS["sound"]
    )
    instruction = instructions[index % len(instructions)]

    prediction = ""
    inference_time = 0.0
    error = None

    for attempt in range(max_retries):
        try:
            result = model.generate_caption(audio_file, instruction)
            raw_prediction = result["output"]
            prediction = sanitize_output(raw_prediction)
            inference_time = result["inference_time"]
            break
        except Exception as e:
            error = str(e)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    result_entry = {
        "id": sample_id,
        "category": category,
        "audio_file": sample["audio_file"],
        "instruction": instruction,
        "prediction": prediction,
        "reference_captions": references,
        "inference_time": inference_time,
    }
    if error and not prediction:
        result_entry["error"] = error
    if "transcript" in sample:
        result_entry["transcript"] = sample["transcript"]

    return result_entry


def run_inference(
    model: AudioCaptionModel,
    samples: List[dict],
    data_dir: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    concurrency: int = 1,
) -> List[dict]:
    """
    Run model inference on all samples.
    Supports concurrent processing with concurrency > 1.

    Returns list of result dicts with prediction, inference_time, etc.
    """
    total = len(samples)

    if concurrency <= 1:
        # Sequential mode (original behavior with full logging)
        results = []
        for i, sample in enumerate(samples):
            print(
                f"\n[{i + 1}/{total}] {sample['id']} "
                f"({sample['category']}, {sample.get('duration_s', '?')}s)"
            )
            instructions = CATEGORY_INSTRUCTIONS.get(
                sample["category"], CATEGORY_INSTRUCTIONS["sound"]
            )
            print(f"  Instruction: {instructions[i % len(instructions)]}")
            print(
                f"  References ({len(sample['reference_captions'])}): "
                f"{sample['reference_captions'][0][:100]}..."
            )

            entry = _infer_single(
                model, sample, i, total, data_dir, max_retries, retry_delay
            )

            if entry.get("error"):
                print(f"  ERROR: {entry['error']}")
            else:
                print(f"  Inference: {entry['inference_time']:.2f}s")
                print(f"  Prediction: {entry['prediction'][:150]}...")

            results.append(entry)
        return results

    # Concurrent mode
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    print(f"  Running with {concurrency} concurrent workers...")
    lock = threading.Lock()
    completed_count = [0]

    results = [None] * total  # Pre-allocate to maintain order

    def _worker(idx_sample):
        idx, sample = idx_sample
        entry = _infer_single(
            model, sample, idx, total, data_dir, max_retries, retry_delay
        )
        with lock:
            completed_count[0] += 1
            done = completed_count[0]
        status = "OK" if not entry.get("error") else "ERR"
        print(
            f"  [{done}/{total}] {entry['id']} ({sample['category']}): "
            f"{status}, {entry['inference_time']:.2f}s"
            f"{' - ' + entry['prediction'][:80] + '...' if status == 'OK' else ''}"
        )
        return idx, entry

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_worker, (i, s)): i
            for i, s in enumerate(samples)
        }
        for future in as_completed(futures):
            idx, entry = future.result()
            results[idx] = entry

    return results


def run_evaluation(
    results: List[dict],
    use_aac_metrics: bool = True,
    use_llm_judge: bool = True,
    judge_model: str = "gpt-4.1",
    concurrency: int = 1,
) -> dict:
    """
    Compute all evaluation metrics on inference results.

    Returns a summary dict with corpus-level and per-sample metrics.
    """
    # Separate successful results
    successful = [r for r in results if "error" not in r or r.get("prediction")]
    predictions = [r["prediction"] for r in successful]
    references_list = [r["reference_captions"] for r in successful]
    categories = [r["category"] for r in successful]
    transcripts = [r.get("transcript", "") for r in successful]

    print(f"\nEvaluating {len(successful)} successful predictions...")

    # 1. Compute metrics
    corpus_scores = {}
    per_sample_scores = [{} for _ in successful]

    if use_aac_metrics and _check_aac_metrics():
        print("  Computing metrics via aac-metrics...")
        try:
            corpus_scores, per_sample_scores = compute_aac_metrics(
                predictions, references_list
            )
            print(f"  aac-metrics computed: {list(corpus_scores.keys())}")
        except Exception as e:
            print(f"  aac-metrics failed: {e}. Falling back to NLTK metrics.")
            corpus_scores, per_sample_scores = compute_fallback_metrics_batch(
                predictions, references_list
            )
    else:
        if use_aac_metrics:
            print("  aac-metrics not available. Using fallback metrics.")
        print("  Computing fallback metrics (METEOR, BLEU, ROUGE-L)...")
        corpus_scores, per_sample_scores = compute_fallback_metrics_batch(
            predictions, references_list
        )

    # Attach per-sample metrics to results
    for j, entry in enumerate(successful):
        entry["metrics"] = per_sample_scores[j]

    # 2. LLM-as-Judge
    if use_llm_judge:
        print("  Setting up LLM judge...")
        llm_client = setup_llm_judge()
        if llm_client:
            n_judge = len(successful)
            if concurrency <= 1:
                # Sequential LLM judge
                print(f"  Running LLM judge ({judge_model})...")
                for j, entry in enumerate(successful):
                    llm_scores = evaluate_with_llm_judge(
                        entry["prediction"],
                        entry["reference_captions"],
                        entry["category"],
                        llm_client,
                        judge_model,
                        entry.get("transcript", ""),
                    )
                    entry["llm_judge"] = llm_scores
                    if llm_scores["overall"] >= 0:
                        print(
                            f"    [{j+1}/{n_judge}] {entry['id']}: "
                            f"overall={llm_scores['overall']:.1f}/10"
                        )
            else:
                # Concurrent LLM judge
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading

                print(f"  Running LLM judge ({judge_model}) with {concurrency} workers...")
                lock = threading.Lock()
                judge_done = [0]

                def _judge_worker(j_entry):
                    j, entry = j_entry
                    scores = evaluate_with_llm_judge(
                        entry["prediction"],
                        entry["reference_captions"],
                        entry["category"],
                        llm_client,
                        judge_model,
                        entry.get("transcript", ""),
                    )
                    with lock:
                        judge_done[0] += 1
                        done = judge_done[0]
                    if scores["overall"] >= 0:
                        print(
                            f"    [{done}/{n_judge}] {entry['id']}: "
                            f"overall={scores['overall']:.1f}/10"
                        )
                    return j, scores

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = {
                        executor.submit(_judge_worker, (j, e)): j
                        for j, e in enumerate(successful)
                    }
                    for future in as_completed(futures):
                        j, scores = future.result()
                        successful[j]["llm_judge"] = scores

            # Aggregate LLM scores
            valid_llm = [
                r["llm_judge"]
                for r in successful
                if r.get("llm_judge", {}).get("overall", -1) >= 0
            ]
            if valid_llm:
                for key in ["accuracy", "completeness",
                           "hallucination", "overall"]:
                    vals = [s[key] for s in valid_llm if s.get(key, -1) >= 0]
                    if vals:
                        corpus_scores[f"llm_{key}"] = sum(vals) / len(vals)

    # 3. Per-category metrics
    per_category = {}
    for cat in ("sound", "music", "speech"):
        cat_entries = [r for r in successful if r["category"] == cat and "metrics" in r]
        if not cat_entries:
            continue
        cat_metrics = {}
        metric_names = list(cat_entries[0]["metrics"].keys())
        for mname in metric_names:
            vals = [
                r["metrics"][mname]
                for r in cat_entries
                if r["metrics"].get(mname, -1) >= 0
            ]
            if vals:
                cat_metrics[mname] = sum(vals) / len(vals)

        # LLM judge per category
        cat_llm = [
            r["llm_judge"]
            for r in cat_entries
            if r.get("llm_judge", {}).get("overall", -1) >= 0
        ]
        if cat_llm:
            for key in ["accuracy", "completeness",
                        "hallucination", "overall"]:
                vals = [s[key] for s in cat_llm if s.get(key, -1) >= 0]
                if vals:
                    cat_metrics[f"llm_{key}"] = sum(vals) / len(vals)

        per_category[cat] = cat_metrics

    return {
        "corpus_scores": corpus_scores,
        "per_category_scores": per_category,
        "results": results,
    }


def print_summary(summary: dict, provider: str, model_id: str) -> None:
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 72)
    print(f"AudioCapBench Evaluation Summary")
    print(f"Provider: {provider} | Model: {model_id}")
    print("=" * 72)

    corpus = summary["corpus_scores"]
    if corpus:
        print(f"\n  Overall ({len([r for r in summary['results'] if 'metrics' in r])} samples):")
        for mname, val in sorted(corpus.items()):
            print(f"    {mname:20s}: {val:.4f}")

    for cat, cat_scores in sorted(summary.get("per_category_scores", {}).items()):
        if cat_scores:
            print(f"\n  {cat.capitalize()}:")
            for mname, val in sorted(cat_scores.items()):
                print(f"    {mname:20s}: {val:.4f}")

    # Count errors
    errors = sum(1 for r in summary["results"] if "error" in r and not r.get("prediction"))
    if errors:
        print(f"\n  Errors: {errors} samples failed inference")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate audio captioning models on AudioCapBench"
    )
    parser.add_argument(
        "--provider", type=str, choices=list_providers(),
        help=f"Model provider: {list_providers()}",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID (uses provider default if not specified)",
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Evaluate all configured models",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/audio_caption",
        help="Directory containing the test set (from build_dataset.py)",
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="Path to metadata.json (default: <data-dir>/metadata.json)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results (default: results/<provider>_<model>.json)",
    )
    parser.add_argument(
        "--credentials", type=str, default=None,
        help="Path to credentials.env file",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--no-aac-metrics", action="store_true",
        help="Disable aac-metrics (use fallback NLTK metrics only)",
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="Disable LLM-as-Judge evaluation",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gpt-4.1",
        help="Model to use as LLM judge (default: gpt-4.1)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per sample on API failure",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["sound", "music", "speech"],
        help="Evaluate only a specific category",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to first N samples (for quick testing)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of concurrent API calls (default: 1, sequential)",
    )
    args = parser.parse_args()

    if not args.provider and not args.all_models:
        parser.error("Either --provider or --all-models is required")

    # Load credentials
    load_credentials(args.credentials)

    # Load metadata
    metadata_path = args.metadata or os.path.join(args.data_dir, "metadata.json")
    print(f"Loading metadata from {metadata_path} ...")
    with open(metadata_path) as f:
        metadata = json.load(f)

    samples = metadata["samples"]

    # Filter by category if requested
    if args.category:
        samples = [s for s in samples if s["category"] == args.category]
        print(f"Filtered to {len(samples)} {args.category} samples")

    # Limit samples for quick testing
    if args.max_samples and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"Limited to first {args.max_samples} samples")

    print(f"Total: {len(samples)} test samples")
    for cat in ("sound", "music", "speech"):
        count = sum(1 for s in samples if s["category"] == cat)
        if count:
            print(f"  {cat}: {count}")

    # Determine which models to evaluate
    if args.all_models:
        providers_to_eval = list_providers()
    else:
        providers_to_eval = [args.provider]

    for provider in providers_to_eval:
        print(f"\n{'=' * 72}")
        print(f"Evaluating with: {provider}")
        print(f"{'=' * 72}")

        model_id = args.model if not args.all_models else None

        try:
            model = create_model(
                provider=provider,
                model_id=model_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"  Failed to initialize {provider}: {e}")
            if args.all_models:
                continue
            else:
                sys.exit(1)

        actual_model_id = model.model_id
        print(f"  Model: {actual_model_id}")

        # Run inference
        results = run_inference(
            model, samples, args.data_dir,
            max_retries=args.max_retries,
            concurrency=args.concurrency,
        )

        # Run evaluation
        summary = run_evaluation(
            results,
            use_aac_metrics=not args.no_aac_metrics,
            use_llm_judge=not args.no_llm_judge,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
        )

        # Add metadata to summary
        summary["provider"] = provider
        summary["model_id"] = actual_model_id
        summary["total_samples"] = len(samples)
        summary["successful"] = sum(
            1 for r in results if "error" not in r or r.get("prediction")
        )
        summary["failed"] = sum(
            1 for r in results if "error" in r and not r.get("prediction")
        )

        # Print summary
        print_summary(summary, provider, actual_model_id)

        # Save results
        output_path = args.output
        if not output_path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            safe_model = actual_model_id.replace("/", "_").replace(".", "_")
            output_path = str(
                results_dir / f"{provider}_{safe_model}_{timestamp}.json"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
