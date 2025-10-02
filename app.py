import streamlit as st
import requests
import base64
import time
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# --- Configuration ---
MODEL_NAME = "google/gemini-2.0-flash-001"
REFINEMENT_MODEL_NAME = "anthropic/claude-sonnet-4.5"
JOURNALS_CSV = "JournalSubset.csv"

# Demo API key - set this to your key, or leave empty to disable demo mode
try:
    DEMO_API_KEY = st.secrets["DEMO_API_KEY"]
except:
    DEMO_API_KEY = ""

MAX_DEMO_USES = 2  # Number of free tries with demo key

# --- Page Configuration ---
st.set_page_config(
    page_title="Journal Matcher",
    page_icon="📚",
    layout="wide"
)

# --- Session State Initialization ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'demo_uses' not in st.session_state:
    st.session_state.demo_uses = 0

# --- API Utilities ---
def process_pdf_with_openrouter(pdf_content, filename, prompt, api_key, model=MODEL_NAME, max_retries=3):
    """Send PDF to OpenRouter API and get the response."""
    
    for attempt in range(max_retries):
        try:
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            data_url = f"data:application/pdf;base64,{pdf_base64}"
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "file",
                                "file": {
                                    "filename": filename,
                                    "file_data": data_url
                                }
                            }
                        ]
                    }
                ],
                "plugins": [{"id": "file-parser", "pdf": {"engine": "native"}}],
                "temperature": 0.3,
                "max_tokens": 16000
            }
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload
            )
            
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                else:
                    return None, f"API Error {response.status_code}: {response.text[:200]}"
            
            result = response.json()
            return result['choices'][0]['message']['content'], None
        
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            else:
                return None, f"Error: {str(e)}"
    
    return None, "Maximum retries exceeded"


def call_openrouter_api(prompt, api_key, model=MODEL_NAME, max_retries=3, max_tokens=50):
    """Generic function to call OpenRouter API."""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'], None
            else:
                time.sleep((attempt + 1) * 2)
        except Exception as e:
            time.sleep((attempt + 1) * 2)

    return None, "API call failed"


def generate_summary(pdf_content, filename, api_key):
    """Generate a concise summary of the PDF."""
    
    summary_prompt = (
        """You will be given a research paper in PDF format. 
        Your task is to produce a **succinct but comprehensive summary** of 400–500 words. 
        The summary should be written in clear academic language and organized under the following sections: 
        (i) Research Question, 
        (ii) Core Methodology, 
        (iii) Main Empirical Findings, and 
        (iv) Academic Subject Area(s). 
        The purpose of this summary is to support journal matching."""
    )
    
    summary, error = process_pdf_with_openrouter(pdf_content, filename, summary_prompt, api_key)
    return summary, error


def get_single_journal_fit(paper_summary, journal_data, api_key):
    """Get fit score for one journal."""
    
    journal_name = journal_data['Name']
    journal_scope = journal_data['Scope']
    journal_subjects = journal_data['Subjects']

    fit_prompt = f"""
You are an expert academic editor. Estimate the likelihood that the research paper summarized below would be accepted for publication in the specified journal, based solely on the alignment between the paper's content and the journal's stated scope and subjects.

PAPER SUMMARY:
---
{paper_summary}
---

JOURNAL DETAILS:
- Name: {journal_name}
- Scope: {journal_scope}
- Subjects: {journal_subjects}

Scoring rules:
- The output is a probability from 0 to 100, where 0 = impossible to publish due to misfit, and 100 = almost certain acceptance if quality is adequate.
- **Scope alignment is the most important criterion.** The paper must directly address the journal's central mission.  
- Broad thematic overlap or vague buzzwords (e.g., "technology," "society") are insufficient.
- Subject overlap contributes **only if** scope alignment is strong.
- Methodological alignment matters only if methods are central to the journal's identity.  
- Anchors:
  - 0–10: Impossible, outside scope entirely.  
  - 11–30: Very unlikely, only tangential overlap.  
  - 31–50: Unlikely, partial but weak alignment.  
  - 51–70: Possible, scope alignment present but not perfect.  
  - 71–85: Likely, strong scope/subject/method fit.  
  - 86–100: Very likely, excellent match across scope, subjects, and methodology.  

Output ONLY a single integer (0–100).  
Do not include explanations, text, punctuation, or formatting.  
"""
    
    score_raw, error = call_openrouter_api(fit_prompt, api_key)
    
    if error:
        return 0
    
    try:
        match = re.search(r'\b(\d{1,3})\b', score_raw)
        if match:
            score = int(match.group(1))
            return min(max(0, score), 100)
        else:
            return 0
    except:
        return 0


def calculate_all_fits_parallel(paper_summary, journals_df, api_key, progress_bar):
    """Calculate journal fit scores in parallel."""
    
    max_workers = max(1, os.cpu_count() - 1)
    fit_scores = [0] * len(journals_df)
    journals_list = journals_df.to_dict('records')
    total_journals = len(journals_list)
    completed_count = 0
    
    def score_journal(index, journal):
        fit_score = get_single_journal_fit(paper_summary, journal, api_key)
        return index, fit_score
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(score_journal, i, journal): i 
            for i, journal in enumerate(journals_list)
        }
        
        for future in as_completed(future_to_index):
            index, score = future.result()
            fit_scores[index] = score
            completed_count += 1
            progress_bar.progress(completed_count / total_journals)
    
    return fit_scores


def refine_top_journals(paper_summary, high_fit_journals_df, api_key):
    """Refine scores for high-fit journals using comparative ranking."""
    
    if len(high_fit_journals_df) == 0:
        return high_fit_journals_df
    
    journals_info = []
    for idx, row in high_fit_journals_df.iterrows():
        journal_entry = f"""
Journal {len(journals_info) + 1}: {row['Name']}
- Initial Fit Score: {row['Fit']}
- Scope: {row['Scope']}
- Subjects: {row['Subjects']}
"""
        journals_info.append(journal_entry)
    
    journals_text = "\n".join(journals_info)
    
    refine_prompt = f"""
You are an expert academic editor. You previously evaluated {len(high_fit_journals_df)} journals independently and all received high fit scores for the research paper summarized below.

PAPER SUMMARY:
---
{paper_summary}
---

JOURNALS TO COMPARE AND RE-RANK:
---
{journals_text}
---

Your task is to provide REFINED fit scores (0-100) for each journal by considering them COMPARATIVELY rather than independently. 

Guidelines:
1. Imagine these journals are competing to be the single best home for this paper.
2. Journals with truly exceptional fit should score 91-100.
3. Journals with strong but not perfect fit should score 81-90.
4. Journals that are decent but clearly weaker should score 71-80.
5. Journals that, upon comparison, appear only marginally suitable can score 61-70.
6. Journals that seem mismatched when compared to others can score below 60.
7. **Scores CAN and SHOULD be reduced** if comparative analysis reveals a weaker match than initial evaluation.
8. Look for subtle differences in scope alignment, methodological fit, and subject area precision.
9. Break ties by considering which journal's readership would find this work most valuable.
10. Avoid giving identical scores unless two journals are truly indistinguishable.

Output format (one line per journal):
Journal 1: [score]
Journal 2: [score]
...
Journal {len(high_fit_journals_df)}: [score]

Output ONLY the numbered list with scores. No explanations or additional text.
"""
    
    refined_scores_raw, error = call_openrouter_api(refine_prompt, api_key, model=REFINEMENT_MODEL_NAME, max_tokens=10000)
    
    if error:
        return high_fit_journals_df
    
    refined_scores = {}
    lines = refined_scores_raw.strip().split('\n')
    
    for line in lines:
        match = re.search(r'Journal\s+(\d+)\s*:\s*(\d{1,3})', line)
        if match:
            journal_num = int(match.group(1))
            score = int(match.group(2))
            refined_scores[journal_num - 1] = min(max(0, score), 100)
    
    if len(refined_scores) == len(high_fit_journals_df):
        high_fit_journals_df = high_fit_journals_df.copy()
        for idx, (df_idx, row) in enumerate(high_fit_journals_df.iterrows()):
            if idx in refined_scores:
                high_fit_journals_df.at[df_idx, 'Fit'] = refined_scores[idx]
    
    return high_fit_journals_df


# --- Main UI ---
st.title("📚 Journal Matcher")
st.markdown("Find the best journals for your research paper using AI-powered analysis.")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    
    # Check if demo mode is enabled
    demo_mode_enabled = DEMO_API_KEY != ""
    
    if demo_mode_enabled:
        remaining_demos = MAX_DEMO_USES - st.session_state.demo_uses
        
        if remaining_demos > 0:
            st.info(f"🎁 **Free Trials Available**: {remaining_demos} of {MAX_DEMO_USES} remaining")
            st.markdown("Try the tool for free, then add your own API key for unlimited use.")
            
            use_demo = st.checkbox("Use free trial", value=True)
            
            if use_demo:
                user_api_key = ""
                st.markdown("---")
                st.markdown("**Or enter your own key:**")
            else:
                use_demo = False
        else:
            st.warning(f"⚠️ You've used all {MAX_DEMO_USES} free trials.")
            st.markdown("Please enter your own OpenRouter API key to continue.")
            use_demo = False
    else:
        use_demo = False
    
    if not use_demo or not demo_mode_enabled or remaining_demos <= 0:
        user_api_key = st.text_input("OpenRouter API Key", type="password", help="Your API key is never stored")
        
        if not demo_mode_enabled or remaining_demos <= 0:
            st.markdown("---")
            st.markdown("### 🔑 How to Get Your API Key")
            st.markdown("""
            <div style="text-align: justify;">
            <b>Step 1: Create Account</b><br>
            Visit <a href="https://openrouter.ai/" target="_blank">OpenRouter.ai</a> and sign up for a free account. You can use your Google account for quick registration.
            <br><br>
            <b>Step 2: Add Credits</b><br>
            Go to your account dashboard and add credits. A minimum of $5 is recommended to start. Credits don't expire.
            <br><br>
            <b>Step 3: Generate API Key</b><br>
            Navigate to the "Keys" section in your dashboard and click "Create Key". Copy the key that starts with "sk-or-v1-".
            <br><br>
            <b>Step 4: Paste Here</b><br>
            Enter your API key in the field above. Your key is never stored and only used for real-time processing.
            <br><br>
            <b>💰 Estimated Cost:</b> $0.10-$0.30 per paper analysis, depending on paper length and database size.
            </div>
            """, unsafe_allow_html=True)
    else:
        user_api_key = ""
    
    st.markdown("---")
    st.markdown("### 📊 Models Used")
    st.markdown("""
    <div style="text-align: justify;">
    <b>Initial Scoring:</b> {}<br>
    Fast and cost-effective model for generating paper summaries and evaluating all journals in the database.
    <br><br>
    <b>Refinement:</b> {}<br>
    Sophisticated model for comparative analysis of top-matching journals, providing nuanced ranking.
    </div>
    """.format(MODEL_NAME, REFINEMENT_MODEL_NAME), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style="text-align: justify;">
    This tool uses artificial intelligence to analyze your research paper and match it with suitable academic journals from our curated database. The analysis happens in two stages: first, a fast evaluation of all journals, then a sophisticated comparative refinement of the best matches.
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("""
<div style="text-align: justify;">
Upload your research paper in PDF format below. The system will analyze your paper's content, methodology, and subject areas to identify the most suitable journals for publication. The process typically takes 2-5 minutes depending on the size of your paper and our journal database.
</div>
""", unsafe_allow_html=True)
st.markdown("")  # Add spacing

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=['pdf'])

# Determine which API key to use
if demo_mode_enabled and use_demo and remaining_demos > 0:
    active_api_key = DEMO_API_KEY
    using_demo = True
else:
    active_api_key = user_api_key
    using_demo = False

# Check if we can run
can_run = active_api_key and uploaded_file

if st.button("🔍 Find Matching Journals", type="primary", disabled=not can_run):
    
    # Increment demo counter if using demo
    if using_demo:
        st.session_state.demo_uses += 1
    
    # Load journals database
    try:
        journals_df = pd.read_csv(JOURNALS_CSV)
    except Exception as e:
        st.error(f"Error loading journals database: {str(e)}")
        st.stop()
    
    # Step 1: Generate Summary
    with st.spinner("📝 Generating paper summary..."):
        pdf_content = uploaded_file.read()
        summary, error = generate_summary(pdf_content, uploaded_file.name, active_api_key)
        
        if error:
            st.error(f"Failed to generate summary: {error}")
            if using_demo:
                st.session_state.demo_uses -= 1  # Don't count failed attempts
            st.stop()
        
        st.session_state.summary = summary
    
    st.success("✅ Summary generated")
    
    with st.expander("📄 View Paper Summary"):
        st.write(summary)
    
    # Step 2: Calculate Initial Fit Scores
    st.markdown("### 🔄 Calculating Initial Fit Scores")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Processing {len(journals_df)} journals...")
    fit_scores = calculate_all_fits_parallel(summary, journals_df, active_api_key, progress_bar)
    journals_df['Fit'] = fit_scores
    
    status_text.text("✅ Initial scoring complete")
    
    # Determine threshold automatically
    count_80 = len(journals_df[journals_df['Fit'] >= 80])
    count_75 = len(journals_df[journals_df['Fit'] >= 75])
    
    if count_80 >= 20:
        threshold = 80
    elif count_75 >= 1:
        threshold = 75
    else:
        st.error("❌ No adequate journals found. No journals scored 75 or above.")
        st.stop()
    
    # Step 3: Refinement
    high_fit_journals = journals_df[journals_df['Fit'] >= threshold].copy()
    
    with st.spinner(f"🎯 Refining {len(high_fit_journals)} top journals..."):
        refined_journals = refine_top_journals(summary, high_fit_journals, active_api_key)
    
    # Sort and prepare results
    final_df = refined_journals.sort_values(by='Fit', ascending=False)
    output_columns = ['Name', 'Publisher', 'JIF', 'Quartile', 'OA', 'Fit']
    final_df = final_df[output_columns]
    
    st.session_state.results = final_df
    
    st.success("✨ Analysis complete!")
    
    # Show reminder about getting own key if using last demo
    if using_demo and st.session_state.demo_uses >= MAX_DEMO_USES:
        st.info("""
        🎉 You've used all your free trials! 
        
        To continue using Journal Matcher, please get your own OpenRouter API key. See the detailed instructions in the sidebar for step-by-step guidance on creating an account and generating your key.
        """)
        st.markdown("---")

# Display Results
if st.session_state.results is not None:
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", len(st.session_state.results))
    with col2:
        st.metric("Highest Fit Score", f"{st.session_state.results['Fit'].max()}")
    with col3:
        st.metric("Average Fit Score", f"{st.session_state.results['Fit'].mean():.1f}")
    
    st.dataframe(
        st.session_state.results,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = st.session_state.results.to_csv(index=False, sep=';')
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="journal_recommendations.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
if demo_mode_enabled:
    st.markdown("""
    <div style="text-align: justify;">
    <i>Demo mode enabled. Try the tool for free with limited trials, then add your own API key for unlimited use. Your API key and uploaded files are never stored on our servers - all processing happens in real-time and data is immediately discarded after analysis.</i>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: justify;">
    <i>Your API key and uploaded files are never stored on our servers. All processing happens in real-time and your data is immediately discarded after analysis to ensure complete privacy and security.</i>
    </div>
    """, unsafe_allow_html=True)
