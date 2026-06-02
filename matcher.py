"""Core logic for JournalFinder.

Talks to OpenRouter to summarize a paper, score journals for fit, and re-rank
the best matches. There's no Streamlit dependency in here, so the logic can be
imported and tested on its own (see test_matcher.py).
"""

from __future__ import annotations

import base64
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import requests

# --- Models (OpenRouter slugs) ---
# One call per paper, and it reads the whole PDF, so it gets a capable model.
SUMMARY_MODEL = "google/gemini-3.5-flash"
# Runs once per journal (~1000+ calls per paper), so it has to be cheap.
SCORING_MODEL = "google/gemini-3.1-flash-lite"
# One call on the shortlist. This is the final ranking, so it gets the best model.
REFINEMENT_MODEL = "anthropic/claude-opus-4.8"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Scoring is network-bound, so we run more threads than CPUs. Higher is faster
# but more likely to hit OpenRouter's rate limits.
DEFAULT_CONCURRENCY = max(4, min(16, (os.cpu_count() or 4) * 2))

# Request timeouts as (connect, read) seconds. The summary reads a whole paper,
# so it's given longer than a single scoring call.
SCORING_TIMEOUT = (10, 60)
SUMMARY_TIMEOUT = (10, 180)
REFINEMENT_TIMEOUT = (10, 180)


def _post(payload: dict, api_key: str, timeout, max_retries: int = 3):
    """POST to OpenRouter with simple exponential backoff.

    Returns (content, error). On success, error is None. On failure, content is
    None and error is a short message.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = "unknown error"
    for attempt in range(max_retries):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_error = f"network error: {exc}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()["choices"][0]["message"]["content"], None
                except (ValueError, KeyError, IndexError) as exc:
                    last_error = f"unexpected response: {exc}"
            else:
                last_error = f"API error {response.status_code}: {response.text[:200]}"
                # A 4xx other than 429 (rate limit) won't fix itself on retry.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return None, last_error
        if attempt < max_retries - 1:
            time.sleep((attempt + 1) * 2)
    return None, last_error


# --- Step 1: summarize the paper ---

SUMMARY_PROMPT = (
    "You will be given a research paper as a PDF. Write a 400-500 word summary "
    "in plain academic language under four headings: (i) Research Question, "
    "(ii) Methods, (iii) Main Findings, and (iv) Subject Area(s). The summary "
    "will be used to match the paper to suitable journals."
)


def generate_summary(pdf_bytes: bytes, filename: str, api_key: str, model: str = SUMMARY_MODEL):
    """Summarize a paper PDF. Returns (summary_text, error)."""
    data_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SUMMARY_PROMPT},
                    {"type": "file", "file": {"filename": filename, "file_data": data_url}},
                ],
            }
        ],
        "plugins": [{"id": "file-parser", "pdf": {"engine": "native"}}],
        "temperature": 0.3,
        "max_tokens": 16000,
    }
    return _post(payload, api_key, SUMMARY_TIMEOUT)


# --- Step 2: score every journal ---

SCORING_PROMPT = """You are an expert academic editor. Estimate the chance that the paper summarized below would be accepted by the journal, based only on how well the paper fits the journal's stated scope and subjects.

PAPER SUMMARY:
---
{summary}
---

JOURNAL:
- Name: {name}
- Scope: {scope}
- Subjects: {subjects}

Rules:
- Output a probability from 0 to 100. 0 = impossible (clear misfit), 100 = near-certain acceptance if the quality is there.
- Scope alignment matters most: the paper must address the journal's central focus.
- Broad overlap or buzzwords ("technology", "society") is not enough.
- Subject overlap counts only when scope alignment is already strong.
- Method fit matters only when methods are central to the journal's identity.
- Anchors: 0-10 outside scope; 11-30 tangential; 31-50 weak; 51-70 plausible; 71-85 strong; 86-100 excellent.

Output ONLY a single integer from 0 to 100. No words, punctuation, or formatting."""


def parse_fit_score(raw: str) -> int:
    """Pull a 0-100 integer out of the model's reply. Returns 0 if none found."""
    if not raw:
        return 0
    match = re.search(r"\b(\d{1,3})\b", raw)
    if not match:
        return 0
    return min(max(int(match.group(1)), 0), 100)


def score_journal(summary: str, journal: dict, api_key: str, model: str = SCORING_MODEL) -> int:
    """Fit score (0-100) for a single journal row."""
    prompt = SCORING_PROMPT.format(
        summary=summary,
        name=journal.get("Name", ""),
        scope=journal.get("Scope", ""),
        subjects=journal.get("Subjects", ""),
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 50,
    }
    raw, error = _post(payload, api_key, SCORING_TIMEOUT)
    if error:
        return 0
    return parse_fit_score(raw)


def score_all_journals(
    summary: str,
    journals: list,
    api_key: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
    max_workers: int = DEFAULT_CONCURRENCY,
    model: str = SCORING_MODEL,
) -> list:
    """Score every journal in parallel.

    `journals` is a list of dict rows. `on_progress(done, total)` is called after
    each one, if given. Returns a list of scores aligned with `journals`.
    """
    total = len(journals)
    scores = [0] * total
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_journal, summary, journal, api_key, model): i
            for i, journal in enumerate(journals)
        }
        for future in as_completed(futures):
            scores[futures[future]] = future.result()
            done += 1
            if on_progress:
                on_progress(done, total)
    return scores


def select_threshold(scores) -> Optional[int]:
    """Pick the shortlist cutoff.

    Use 80 when there are enough strong matches to compare (>= 20), otherwise 75
    if there's at least one, otherwise None (nothing worth showing).
    """
    scores = list(scores)
    if sum(s >= 80 for s in scores) >= 20:
        return 80
    if sum(s >= 75 for s in scores) >= 1:
        return 75
    return None


# --- Step 3: re-rank the shortlist ---

REFINEMENT_PROMPT = """You are an expert academic editor. The {n} journals below all scored highly for the paper summarized here when judged one at a time. Re-score them 0-100 by comparing them against each other.

PAPER SUMMARY:
---
{summary}
---

JOURNALS:
---
{journals}
---

Guidance:
- Treat the journals as competing to be the single best home for this paper.
- 91-100 exceptional fit; 81-90 strong; 71-80 decent but clearly weaker; 61-70 marginal; below 60 mismatched in comparison.
- Lower a score if the comparison reveals a weaker match than it first seemed.
- Weigh scope alignment, method fit, and subject precision; break ties by which readership would value the work most.
- Avoid identical scores unless two journals really are indistinguishable.

Output one line per journal, nothing else:
Journal 1: <score>
Journal 2: <score>
...
Journal {n}: <score>"""


def parse_refined_scores(raw: str, n: int) -> dict:
    """Parse 'Journal K: score' lines into {index0: score}, ignoring out-of-range K."""
    scores = {}
    for line in (raw or "").splitlines():
        match = re.search(r"Journal\s+(\d+)\s*:\s*(\d{1,3})", line)
        if match:
            k = int(match.group(1)) - 1
            if 0 <= k < n:
                scores[k] = min(max(int(match.group(2)), 0), 100)
    return scores


def refine_scores(summary: str, shortlist: list, api_key: str, model: str = REFINEMENT_MODEL) -> list:
    """Re-rank the shortlist comparatively.

    `shortlist` is a list of dict rows that already carry a 'Fit' score. Returns a
    list of refined scores aligned with `shortlist`. If refinement fails or the
    model doesn't return a score for every journal, the original fits are kept.
    """
    n = len(shortlist)
    original = [int(j.get("Fit", 0)) for j in shortlist]
    if n == 0:
        return original
    journals_block = "\n".join(
        f"Journal {i + 1}: {j.get('Name', '')}\n"
        f"- Initial Fit: {j.get('Fit', 0)}\n"
        f"- Scope: {j.get('Scope', '')}\n"
        f"- Subjects: {j.get('Subjects', '')}\n"
        for i, j in enumerate(shortlist)
    )
    prompt = REFINEMENT_PROMPT.format(n=n, summary=summary, journals=journals_block)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10000,
    }
    raw, error = _post(payload, api_key, REFINEMENT_TIMEOUT)
    if error:
        return original
    refined = parse_refined_scores(raw, n)
    if len(refined) != n:
        return original
    return [refined[i] for i in range(n)]
