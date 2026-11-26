# 📰 Financial News Sentiment Analyzer

A classroom-friendly Streamlit application demonstrating **AI, sentiment analysis, Generative AI, and finance** through financial-news headlines.

## Features

- Positive / Neutral / Negative classification
- Approximate sentiment probability distribution
- Bullish / Bearish / Neutral market-tone inference
- Strength estimate (Low / Moderate / High)
- Company, sector, and event category heuristics
- Explanation of why investors might react
- Potential investor implications and risk factors
- Rule-based analysis that works without an API key
- Optional OpenAI-powered LLM analysis
- Batch analysis of multiple headlines
- Sentiment composition visualization
- CSV and JSON downloads
- Sample headlines for classroom demonstrations

> **Disclaimer:** This is an educational demonstration, not financial advice, and it does not forecast stock prices or recommend trades. Rule-based confidence percentages are illustrative, not calibrated model probabilities.

---

## 1. Project files

```text
financial_news_sentiment_analyzer/
├── app.py
├── requirements.txt
└── README.md
```

## 2. Installation

Recommended: Python 3.10 or later.

### Create and activate a virtual environment

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the application

From the project folder:

```bash
streamlit run app.py
```

Streamlit will show a local URL (usually `http://localhost:8501`) in your browser.

## 4. Optional: Enable LLM-powered analysis

The app works without an API key in **Rule-based (offline)** mode.

To use the OpenAI API, select **OpenAI API** in the sidebar and supply a valid key. For safer configuration, set it as an environment variable rather than pasting it into the interface.

### Windows PowerShell
```powershell
$env:OPENAI_API_KEY="your-api-key"
streamlit run app.py
```

### macOS / Linux
```bash
export OPENAI_API_KEY="your-api-key"
streamlit run app.py
```

You may also use Streamlit secrets. Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-api-key"
```

**Do not commit or share API keys.** Add `.streamlit/secrets.toml` to `.gitignore` if using this option. In the app sidebar, the model can be selected (default: `gpt-4o-mini`). Confirm that the chosen model is available for your API account.

LLM mode sends the submitted headline to the configured API provider for analysis. Avoid submitting confidential, unpublished, or sensitive information.

## 5. Classroom demonstration

Suggested example headlines:

### Positive
> Reliance Industries reports strong quarterly earnings and announces major investment in renewable energy.

Expected classroom discussion: earnings performance, investment, future growth expectations, and potential positive sentiment.

### Negative
> Company faces regulatory investigation after disappointing quarterly results.

Expected classroom discussion: regulatory risk, uncertainty, earnings disappointment, and possible bearish sentiment.

### Mixed
> Revenue rises sharply, but profit margins decline and management warns of higher costs.

Expected classroom discussion: contradictory signals, why sentiment may be mixed, and the importance of understanding the details.

### Neutral
> Company announces appointment of a new chief operating officer.

Expected classroom discussion: a factual event whose financial impact may depend on the executive's track record and strategic role.

## 6. Suggested learning outcomes

After the demonstration, students should be able to:

1. Distinguish sentiment classification from price forecasting.
2. Explain why investors may react to financial news.
3. Compare a lexical/rule-based method with a generative AI method.
4. Interpret confidence scores critically.
5. Identify risks of headline-only analysis.
6. Discuss the importance of market expectations and information already priced in.

## 7. How the app works

### Rule-based mode
A finance-oriented keyword lexicon scores positive and negative language. The score is converted into a simple sentiment label and illustrative confidence distribution. A small set of heuristics also estimates company, sector, and event type.

### OpenAI API mode
The selected language model receives the headline and returns a structured JSON response containing sentiment, probability scores, market tone, explanation, investor implications, and limitations. If the API call fails, the application displays a warning and falls back to rule-based analysis.

### Batch mode
Submit multiple headlines, one per line. The app displays a table, sentiment composition chart, and downloadable CSV/JSON files.

## 8. Limitations

- Headlines omit important context from the full article.
- Sentiment and market direction are not equivalent.
- Market response depends on surprise relative to expectations, valuation, liquidity, macroeconomic conditions, and prior positioning.
- The rule-based mode has a limited vocabulary and may misinterpret negation or complex statements.
- Rule-based confidence is heuristic and not calibrated on a labeled dataset.
- Company and sector detection are simplified heuristics, not production-grade named entity recognition.
- LLM explanations can be plausible but inaccurate.
- The app does not retrieve live news, financial data, prices, or company filings.

## 9. Possible extensions for student projects

- Add a live-news feed using a news provider API.
- Add ticker/company entity extraction with spaCy or FinBERT.
- Compare sentiment with historical stock returns.
- Use FinBERT for finance-specific pretrained sentiment.
- Add sentiment over time and a sector-level dashboard.
- Compare sentiment across multiple LLMs.
- Add a student-built labeled dataset and evaluate precision, recall, F1, and confusion matrix.
- Add a “headline vs full article” comparison.
- Create a simulated portfolio experiment (not real investment recommendations).
- Add multilingual headline analysis.

## 10. Troubleshooting

**`streamlit` command not found**  
Activate the virtual environment and reinstall dependencies. Alternatively, run:

```bash
python -m streamlit run app.py
```

**OpenAI analysis fails**  
Verify your key, API account access, internet connection, selected model availability, and API billing/usage limits. Rule-based mode will remain available.

**Plots do not display**  
Update Streamlit and Plotly:

```bash
pip install --upgrade streamlit plotly
```

---

## License 

This repository is made publicly available for educational, academic, and classroom use. Permission is granted to access and use the code for learning, teaching, and academic demonstration purposes. Redistribution, commercial use, modification for public distribution, or incorporation into other projects requires prior written permission from the copyright holder, unless otherwise specified. This repository is provided "as is", without warranty of any kind. The author assumes no responsibility for any loss or damage arising from its use. For other permissions or licensing inquiries, please contact the author.
