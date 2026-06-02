# JournalFinder

Upload a research paper (PDF) and get a ranked list of journals that fit its scope.
It reads the paper, scores every journal in `JournalSubset.csv` for fit, then
re-ranks the best matches with a stronger model.

## How it works

1. A summary of the paper is generated from the PDF.
2. Each journal in the database is scored 0–100 for fit against that summary.
3. The top matches are re-ranked by comparing them against each other.
4. You get a sorted table, downloadable as CSV.

Everything runs through [OpenRouter](https://openrouter.ai/), so one API key covers all three steps.

## Models

| Step | Model | Why |
|------|-------|-----|
| Summary | `google/gemini-3.5-flash` | One call per paper; reads the full PDF. |
| Scoring | `google/gemini-3.1-flash-lite` | Runs ~1,000+ times per paper, so it has to be cheap and fast. |
| Re-ranking | `anthropic/claude-opus-4.8` | One call on the shortlist — the final ranking, so it gets the best model. |

Change any of these at the top of `matcher.py`. Rough cost is about **$0.40 per paper**,
almost all of it from the scoring step. Swapping the scoring model for an even
cheaper one (e.g. `google/gemini-2.5-flash-lite`) is a one-line change.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501, paste your OpenRouter key in the sidebar, upload a PDF,
and click **Find journals**.

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io/), create a new app from the repo.
3. Set the main file to `streamlit_app.py`.
4. Optional: to offer free trial runs on your own key, add a secret named `DEMO_API_KEY`
   in the app settings. Each trial run is billed to that key.

## The journal database

`JournalSubset.csv` holds the journals to match against, with columns
`Name, Publisher, JIF, Quartile, OA, Scope, Subjects`. Add or remove rows to change
what gets searched — only `Name`, `Scope`, and `Subjects` affect scoring; the rest
are shown in the results.

## Tests

```bash
python test_matcher.py    # or: pytest
```

These cover the score parsing and shortlist logic and don't make any network calls.

## Notes

- Your API key and uploaded PDF aren't stored; they're used for the run and then dropped.
- The free-trial counter is per browser session, so it's a convenience, not a hard limit.
