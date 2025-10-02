# Journal Matcher

AI-powered tool to find the best academic journals for your research paper.

## Features

- 📝 Automatic paper summary generation
- 🔍 Intelligent journal matching using dual-model AI analysis
- 📊 Comparative refinement of top journal matches
- 🚀 Easy-to-use web interface

## How It Works

1. **Summary Generation**: AI reads your PDF and creates a comprehensive summary
2. **Initial Scoring**: Fast model evaluates all journals in the database
3. **Smart Refinement**: Sophisticated model performs comparative analysis on top matches
4. **Results**: Ranked list of suitable journals with fit scores

## Models Used

- **Initial Scoring**: Google Gemini 2.0 Flash (fast and cost-effective)
- **Refinement**: Claude Sonnet 4.5 (sophisticated comparative analysis)

## Deployment Instructions

### Prerequisites

- GitHub account
- Streamlit Cloud account (free - sign up at https://share.streamlit.io/)
- OpenRouter API key (users provide their own)

### Setup Steps

1. **Create Private GitHub Repository**
   - Go to GitHub and create a new private repository
   - Name it something like `journal-matcher`

2. **Upload Files**
   - `app.py` (the Streamlit application)
   - `requirements.txt` (Python dependencies)
   - `JournalSubset.csv` (your journals database)
   - `.gitignore` (excluded files)
   - `README.md` (this file)

3. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your private repository
   - Main file path: `app.py`
   - Click "Deploy"
   - Wait 2-3 minutes for deployment

4. **Share with Colleagues**
   - Copy the deployed app URL (e.g., `https://your-app.streamlit.app`)
   - Share with colleagues
   - They'll need their own OpenRouter API key to use the app

### Local Testing (Optional)

Before deploying, you can test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## User Instructions

### For Your Colleagues

1. Visit the app URL
2. Enter your OpenRouter API key in the sidebar
3. Upload your research paper (PDF)
4. Click "Find Matching Journals"
5. Wait for analysis (usually 2-5 minutes depending on database size)
6. Review results and download recommendations

### Getting an OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up for an account
3. Add credits to your account
4. Generate an API key from the dashboard

**Estimated Cost**: Approximately $0.10-$0.30 per paper analysis (depends on paper length and number of journals)

## Privacy & Security

- API keys are never stored or logged
- Uploaded PDFs are processed in memory only
- No data is saved between sessions
- All processing happens in real-time

## Troubleshooting

### "No adequate journals found"
- This means no journals scored 75 or above
- The paper may be too specialized or outside the database scope
- Try a different paper or expand the journal database

### API Errors
- Check that your API key is valid
- Ensure you have credits in your OpenRouter account
- Verify internet connection

### Slow Performance
- Large PDFs take longer to process
- Database with many journals increases processing time
- This is normal - be patient!

## Support

For issues or questions, contact the repository owner.

## License

Private use only - not for public distribution.
