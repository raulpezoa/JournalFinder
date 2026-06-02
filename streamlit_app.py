"""JournalFinder web app.

Streamlit Cloud runs this file by default. app.py is a thin wrapper that calls
main() here, so either entry point works. All the matching logic lives in
matcher.py.
"""

import pandas as pd
import streamlit as st

import matcher

JOURNALS_CSV = "JournalSubset.csv"
RESULT_COLUMNS = ["Name", "Publisher", "JIF", "Quartile", "OA", "Fit"]
MAX_DEMO_USES = 2


def get_demo_key() -> str:
    """Owner-supplied key for free trial runs, or "" if none is configured."""
    try:
        return st.secrets.get("DEMO_API_KEY", "")
    except Exception:
        return ""


@st.cache_data
def load_journals(path: str = JOURNALS_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def run_match(pdf_bytes: bytes, filename: str, api_key: str):
    """Run the full pipeline with on-screen feedback. Returns a results
    DataFrame, or None if the paper couldn't be summarized or nothing matched."""
    journals = load_journals()

    with st.spinner("Reading the paper..."):
        summary, error = matcher.generate_summary(pdf_bytes, filename, api_key)
    if error:
        st.error(f"Couldn't read the PDF: {error}")
        return None
    st.session_state.summary = summary

    st.write(f"Scoring {len(journals)} journals...")
    progress = st.progress(0.0)
    scores = matcher.score_all_journals(
        summary,
        journals.to_dict("records"),
        api_key,
        on_progress=lambda done, total: progress.progress(done / total),
    )
    progress.empty()
    journals = journals.assign(Fit=scores)

    threshold = matcher.select_threshold(scores)
    if threshold is None:
        st.warning(
            "No strong matches — nothing scored 75 or above. "
            "The paper may sit outside this set of journals."
        )
        return None

    shortlist = journals[journals["Fit"] >= threshold].copy()
    with st.spinner(f"Re-ranking the top {len(shortlist)} matches..."):
        shortlist["Fit"] = matcher.refine_scores(summary, shortlist.to_dict("records"), api_key)

    return shortlist.sort_values("Fit", ascending=False)[RESULT_COLUMNS].reset_index(drop=True)


def render_sidebar():
    """Draw the sidebar. Returns (api_key, using_demo)."""
    demo_key = get_demo_key()
    st.session_state.setdefault("demo_uses", 0)
    demo_left = MAX_DEMO_USES - st.session_state.demo_uses

    with st.sidebar:
        st.subheader("API key")
        api_key = ""
        using_demo = False

        if demo_key and demo_left > 0:
            using_demo = st.checkbox(f"Use a free trial run ({demo_left} left)", value=True)
            if using_demo:
                api_key = demo_key
            else:
                api_key = st.text_input("OpenRouter API key", type="password")
        else:
            if demo_key:
                st.caption("Free trials used up. Add your own key to keep going.")
            api_key = st.text_input("OpenRouter API key", type="password")

        with st.expander("How to get a key"):
            st.markdown(
                "1. Sign up at [openrouter.ai](https://openrouter.ai/)\n"
                "2. Add a few dollars of credit\n"
                "3. Create a key under **Keys** and paste it above\n\n"
                "Each paper costs roughly **$0.40**."
            )

        st.subheader("Models")
        st.caption(f"Summary — `{matcher.SUMMARY_MODEL}`")
        st.caption(f"Scoring — `{matcher.SCORING_MODEL}`")
        st.caption(f"Re-ranking — `{matcher.REFINEMENT_MODEL}`")

    return api_key, using_demo


def render_results(results: pd.DataFrame):
    st.subheader("Matches")

    c1, c2, c3 = st.columns(3)
    c1.metric("Journals", len(results))
    c2.metric("Best fit", int(results["Fit"].max()))
    c3.metric("Average fit", f"{results['Fit'].mean():.0f}")

    if st.session_state.get("summary"):
        with st.expander("Paper summary"):
            st.write(st.session_state.summary)

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn("Journal", width="large"),
            "Publisher": st.column_config.TextColumn("Publisher"),
            "JIF": st.column_config.NumberColumn("JIF", format="%.1f"),
            "Quartile": st.column_config.TextColumn("Q"),
            "OA": st.column_config.TextColumn("Open access"),
            "Fit": st.column_config.ProgressColumn("Fit", min_value=0, max_value=100, format="%d"),
        },
    )
    st.download_button(
        "Download as CSV",
        results.to_csv(index=False, sep=";"),
        file_name="journal_matches.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="JournalFinder", page_icon="📑", layout="wide")
    st.session_state.setdefault("results", None)

    st.title("JournalFinder")
    st.write("Upload a paper and get a ranked list of journals that fit its scope.")

    api_key, using_demo = render_sidebar()
    uploaded = st.file_uploader("Paper (PDF)", type=["pdf"])

    if st.button("Find journals", type="primary", disabled=not (api_key and uploaded)):
        results = run_match(uploaded.getvalue(), uploaded.name, api_key)
        if results is not None and using_demo:
            st.session_state.demo_uses += 1  # only a successful run uses a trial
        st.session_state.results = results

    if st.session_state.results is not None:
        render_results(st.session_state.results)

    st.divider()
    st.caption(
        "Your API key and uploaded PDF aren't stored — they're used for the run and then dropped."
    )


if __name__ == "__main__":
    main()
