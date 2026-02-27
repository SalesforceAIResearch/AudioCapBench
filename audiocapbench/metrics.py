#!/usr/bin/env python3
"""
Evaluation metrics for audio captioning.

Primary: aac-metrics (CIDEr-D, SPICE, SPIDEr, FENSE, METEOR, BLEU, ROUGE-L)
Fallback: NLTK + rouge-score (if aac-metrics/Java unavailable)
Optional: LLM-as-Judge via OpenAI-compatible API
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ===================================================================
# aac-metrics (primary, comprehensive)
# ===================================================================

_AAC_METRICS_AVAILABLE = None


def _check_aac_metrics() -> bool:
    """Check if aac-metrics is installed and functional."""
    global _AAC_METRICS_AVAILABLE
    if _AAC_METRICS_AVAILABLE is not None:
        return _AAC_METRICS_AVAILABLE
    try:
        import aac_metrics
        _AAC_METRICS_AVAILABLE = True
    except ImportError:
        _AAC_METRICS_AVAILABLE = False
    return _AAC_METRICS_AVAILABLE


def compute_aac_metrics(
    predictions: List[str],
    references_list: List[List[str]],
    metrics: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Compute audio captioning metrics using aac-metrics.

    Args:
        predictions: List of predicted captions.
        references_list: List of lists of reference captions per sample.
        metrics: Which metrics to compute. If None, uses DCASE-standard set.

    Returns:
        Tuple of (corpus_scores, per_sample_scores).
        corpus_scores: Dict mapping metric name -> corpus-level score.
        per_sample_scores: List of dicts, one per sample.
    """
    from aac_metrics import evaluate

    # aac-metrics expects mult_references as list of list of str
    # and candidates as list of str
    if metrics:
        corpus_scores, sent_scores = evaluate(
            predictions, references_list, metrics=metrics
        )
    else:
        # Default: standard captioning metrics
        corpus_scores, sent_scores = evaluate(
            predictions, references_list,
        )

    # Convert to plain dicts
    corpus_dict = {}
    for k, v in corpus_scores.items():
        if hasattr(v, "item"):
            corpus_dict[k] = v.item()
        else:
            corpus_dict[k] = float(v)

    per_sample = []
    n = len(predictions)
    for i in range(n):
        sample_dict = {}
        for k, v in sent_scores.items():
            val = v[i]
            if hasattr(val, "item"):
                sample_dict[k] = val.item()
            else:
                sample_dict[k] = float(val)
        per_sample.append(sample_dict)

    return corpus_dict, per_sample


# ===================================================================
# Fallback metrics (NLTK + rouge-score)
# ===================================================================

def compute_meteor(prediction: str, references: List[str]) -> float:
    """Compute METEOR score (max across references)."""
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        pred_tokens = prediction.lower().split()
        scores = []
        for ref in references:
            ref_tokens = ref.lower().split()
            score = nltk_meteor([ref_tokens], pred_tokens)
            scores.append(score)
        return max(scores) if scores else 0.0
    except ImportError:
        return -1.0


def compute_bleu(prediction: str, references: List[str], n: int = 4) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-n with smoothing."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1

        pred_tokens = prediction.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in references]

        results = {}
        for i in range(1, n + 1):
            weights = tuple([1.0 / i] * i + [0.0] * (4 - i))
            score = sentence_bleu(
                ref_tokens_list, pred_tokens,
                weights=weights, smoothing_function=smooth,
            )
            results[f"bleu_{i}"] = score
        return results
    except ImportError:
        return {f"bleu_{i}": -1.0 for i in range(1, n + 1)}


def compute_rouge_l(prediction: str, references: List[str]) -> float:
    """Compute ROUGE-L F1 (max across references)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for ref in references:
            result = scorer.score(ref, prediction)
            scores.append(result["rougeL"].fmeasure)
        return max(scores) if scores else 0.0
    except ImportError:
        return _rouge_l_fallback(prediction, references)


def _rouge_l_fallback(prediction: str, references: List[str]) -> float:
    """Simple LCS-based ROUGE-L without external dependencies."""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    pred_tokens = prediction.lower().split()
    scores = []
    for ref in references:
        ref_tokens = ref.lower().split()
        lcs = lcs_length(pred_tokens, ref_tokens)
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)
    return max(scores) if scores else 0.0


def compute_fallback_metrics(
    prediction: str, references: List[str]
) -> Dict[str, float]:
    """Compute all fallback metrics for a single prediction."""
    metrics = {}
    metrics["meteor"] = compute_meteor(prediction, references)
    metrics.update(compute_bleu(prediction, references))
    metrics["rouge_l"] = compute_rouge_l(prediction, references)
    return metrics


def compute_fallback_metrics_batch(
    predictions: List[str],
    references_list: List[List[str]],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Compute fallback metrics for a batch. Returns same format as
    compute_aac_metrics for consistency.
    """
    per_sample = []
    for pred, refs in zip(predictions, references_list):
        per_sample.append(compute_fallback_metrics(pred, refs))

    # Aggregate corpus-level scores
    corpus = {}
    if per_sample:
        metric_names = list(per_sample[0].keys())
        for mname in metric_names:
            vals = [s[mname] for s in per_sample if s[mname] >= 0]
            if vals:
                corpus[mname] = sum(vals) / len(vals)
    return corpus, per_sample


# ===================================================================
# LLM-as-Judge
# ===================================================================

def setup_llm_judge(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Any]:
    """Setup OpenAI client for LLM judge evaluation."""
    try:
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("  LLM Judge: OPENAI_API_KEY not set. Skipping.")
            return None

        url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Salesforce gateway: pass key via X-Api-Key header
        if url and "gateway.salesforceresearch.ai" in url:
            client = OpenAI(
                api_key="dummy",
                base_url=url,
                default_headers={"X-Api-Key": key},
            )
        else:
            kwargs = {}
            if url:
                kwargs["base_url"] = url
            client = OpenAI(api_key=key, **kwargs)

        return client
    except ImportError:
        print("  LLM Judge: openai package not installed. Skipping.")
        return None
    except Exception as e:
        print(f"  LLM Judge: setup failed: {e}. Skipping.")
        return None


def evaluate_with_llm_judge(
    prediction: str,
    references: List[str],
    category: str,
    llm_client: Any,
    judge_model: str = "gpt-4.1",
    transcript: str = "",
) -> Dict[str, Any]:
    """
    Use an LLM judge to score a predicted caption against references.

    Returns dict with: accuracy, completeness, fluency, overall, reasoning.
    """
    refs_text = "\n".join(f"  - {r}" for r in references)

    category_guidance = {
        "music": (
            "For music, evaluate whether the prediction correctly identifies: "
            "genre, instrumentation, tempo, mood/atmosphere, and any vocal characteristics."
        ),
        "speech": (
            "For speech, evaluate whether the prediction correctly describes: "
            "speaker characteristics (gender, age, accent), emotional tone, "
            "speaking style, and the content of what is being said. "
            "The reference may include both a transcript and an emotional description - "
            "evaluate how well the prediction captures both aspects."
        ),
    }.get(category, (
        "For general/environmental sound, evaluate whether the prediction "
        "correctly identifies: sound sources, events, acoustic environment, "
        "and temporal patterns."
    ))

    if category == "speech" and transcript:
        category_guidance += f'\nActual transcript of the speech: "{transcript}"'

    prompt = f"""You are an expert evaluator for audio captioning systems.

Given the ground-truth reference captions and a model's predicted caption for an audio clip,
score the prediction on the following criteria (each on a scale of 0 to 10):

1. **Accuracy** (0-10): Does the prediction correctly describe the same audio content as the references?
   Are the key sound sources, events, or attributes correct? Note: the prediction may use different
   wording than the references - focus on whether the semantic content is correct, not exact word matches.
2. **Completeness** (0-10): Does the prediction cover the main elements mentioned in the references?
   Are important details missing? A prediction that captures the most salient elements should score
   highly even if it misses minor details.
3. **Hallucination** (0-10): Does the prediction ONLY describe sounds/events that are actually
   supported by the references? 10 = no hallucination (everything described matches the references),
   0 = heavy hallucination (the prediction invents sounds, events, or attributes not present in the
   references). Penalize any fabricated content, even if the prediction also contains correct elements.

{category_guidance}

Reference captions (ground-truth):
{refs_text}

Model prediction:
  "{prediction}"

Important:
- If the prediction is empty, all scores should be 0.
- Scores must be integers from 0 to 10.
- Focus on semantic similarity, not surface-level word overlap.

Respond with ONLY a JSON object, no other text:
{{"accuracy": <int 0-10>, "completeness": <int 0-10>, "hallucination": <int 0-10>, "reasoning": "<1-2 sentence explanation>"}}"""

    try:
        response = llm_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()

        json_str = raw
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*(.*?)```", json_str, re.DOTALL)
            json_str = match.group(1).strip() if match else raw

        scores = json.loads(json_str)
        accuracy = max(0, min(10, float(scores.get("accuracy", 0))))
        completeness = max(0, min(10, float(scores.get("completeness", 0))))
        hallucination = max(0, min(10, float(scores.get("hallucination", 0))))
        overall = (accuracy + completeness + hallucination) / 3.0

        return {
            "accuracy": accuracy,
            "completeness": completeness,
            "hallucination": hallucination,
            "overall": round(overall, 2),
            "reasoning": scores.get("reasoning", ""),
        }
    except Exception as e:
        print(f"    LLM judge error: {e}")
        return {
            "accuracy": -1, "completeness": -1,
            "hallucination": -1, "overall": -1,
            "reasoning": f"Error: {e}",
        }


def evaluate_llm_judge_batch(
    predictions: List[str],
    references_list: List[List[str]],
    categories: List[str],
    llm_client: Any,
    judge_model: str = "gpt-4.1",
    transcripts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run LLM judge on a batch of predictions."""
    results = []
    for i, (pred, refs, cat) in enumerate(
        zip(predictions, references_list, categories)
    ):
        transcript = transcripts[i] if transcripts else ""
        result = evaluate_with_llm_judge(
            pred, refs, cat, llm_client, judge_model, transcript
        )
        results.append(result)
    return results
