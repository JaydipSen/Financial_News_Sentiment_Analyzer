"""
Financial News Sentiment Analyzer
Classroom-ready Streamlit app for finance and generative AI demonstrations.

Modes:
- OpenAI API analysis (optional, configured through OPENAI_API_KEY)
- Rule-based financial sentiment fallback (works offline)

Disclaimer: Educational analysis only, not investment advice.
"""
import os
import json
import re
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

# OpenAI is optional; app falls back to rule-based analysis if unavailable.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Styling ---------------------------------------
st.markdown("""
<style>
.main .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(120,120,120,.18); padding: 12px; border-radius: 12px;}
.small-note {font-size: .88rem; color: #777;}
</style>
""", unsafe_allow_html=True)

# ------------------------------ Core logic ------------------------------------
POSITIVE_TERMS = {
    "strong": 2, "stronger": 2, "growth": 2, "grows": 2, "grew": 2,
    "profit": 2, "profits": 2, "profitable": 2, "earnings": 2, "beat": 3,
    "beats": 3, "record": 2, "surge": 3, "surges": 3, "rally": 3,
    "rallies": 3, "upgrade": 2, "upgraded": 2, "outperform": 2,
    "outperforms": 2, "positive": 2, "success": 2, "successful": 2,
    "investment": 1, "invest": 1, "expansion": 2, "expands": 2,
    "dividend": 2, "buyback": 2, "acquisition": 1, "innovation": 2,
    "renewable": 1, "recovery": 2, "recover": 2, "breakthrough": 3,
    "approval": 2, "approved": 2, "launch": 1, "launched": 1,
    "deal": 1, "contract": 2, "contracts": 2, "demand": 1,
    "raises": 2, "raised": 2, "guidance": 1, "optimistic": 2,
    "cash": 1, "resilient": 2, "robust": 2, "wins": 2, "won": 2,
}
NEGATIVE_TERMS = {
    "weak": 2, "weaker": 2, "decline": 2, "declines": 2, "declining": 2,
    "loss": 3, "losses": 3, "unprofitable": 3, "miss": 3, "misses": 3,
    "missed": 3, "disappointing": 3, "disappoint": 2, "disappointment": 2,
    "crash": 4, "crashes": 4, "plunge": 4, "plunges": 4, "plunged": 4,
    "fall": 2, "falls": 2, "fell": 2, "drop": 2, "drops": 2, "dropped": 2,
    "downgrade": 2, "downgraded": 2, "underperform": 2, "negative": 2,
    "debt": 1, "default": 4, "defaults": 4, "bankruptcy": 4,
    "investigation": 2, "investigated": 2, "regulatory": 1, "fraud": 4,
    "scandal": 3, "lawsuit": 2, "litigation": 2, "fine": 2, "penalty": 2,
    "penalties": 2, "recall": 2, "recalled": 2, "layoff": 2, "layoffs": 2,
    "cuts": 2, "cut": 1, "risk": 1, "risks": 1, "uncertainty": 2,
    "uncertain": 1, "concern": 1, "concerns": 1, "warning": 2,
    "crisis": 3, "slowdown": 2, "recession": 3, "inflation": 1,
    "cost": 1, "costs": 1, "loss-making": 3, "halt": 2, "halted": 2,
    "resigns": 2, "resigned": 2, "shutdown": 3, "breach": 3, "attack": 2,
}
NEGATION = {"not", "no", "never", "without", "hardly", "fails", "failed"}
NEUTRAL_TERMS = {"announces", "announcement", "reports", "reported", "says", "said", "plans", "plan", "appoints", "appointed"}

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())

def rule_analysis(headline: str) -> Dict:
    words = tokenize(headline)
    pos, neg = 0, 0
    evidence_pos, evidence_neg = [], []
    for i, word in enumerate(words):
        prev = words[max(0, i-3):i]
        is_negated = any(w in NEGATION for w in prev)
        if word in POSITIVE_TERMS:
            weight = POSITIVE_TERMS[word]
            if is_negated:
                neg += weight
                evidence_neg.append(word + " (negated)")
            else:
                pos += weight
                evidence_pos.append(word)
        if word in NEGATIVE_TERMS:
            weight = NEGATIVE_TERMS[word]
            if is_negated:
                pos += max(1, weight - 1)
                evidence_pos.append(word + " (negated)")
            else:
                neg += weight
                evidence_neg.append(word)

    # A headline without directional cues should remain neutral.
    total = pos + neg
    if total == 0:
        label = "Neutral"
        scores = {"Positive": 15, "Neutral": 70, "Negative": 15}
        impact = "Neutral / Unclear"
        impact_direction = "Neutral"
        strength = "Low"
    else:
        net = pos - neg
        label = "Positive" if net > 0 else "Negative" if net < 0 else "Neutral"
        # Smooth probabilities, with a confidence ceiling for a lexical heuristic.
        directional = min(0.84, 0.53 + (abs(net) / max(total, 1)) * 0.27 + min(total, 8)*0.015)
        if label == "Positive":
            scores = {"Positive": round(directional*100), "Neutral": round((1-directional)*55*100), "Negative": 0}
            scores["Negative"] = 100 - scores["Positive"] - scores["Neutral"]
            impact = "Bullish"
            impact_direction = "Bullish"
        elif label == "Negative":
            scores = {"Negative": round(directional*100), "Neutral": round((1-directional)*55*100), "Positive": 0}
            scores["Positive"] = 100 - scores["Negative"] - scores["Neutral"]
            impact = "Bearish"
            impact_direction = "Bearish"
        else:
            scores = {"Positive": 20, "Neutral": 60, "Negative": 20}
            impact = "Neutral / Mixed"
            impact_direction = "Neutral"
        strength = "High" if abs(net) >= 5 else "Moderate" if abs(net) >= 2 else "Low"
    # Ensure integer score totals exactly 100.
    scores = {k: max(0, int(v)) for k, v in scores.items()}
    scores["Neutral"] += 100 - sum(scores.values())
    company = extract_company(headline)
    sector = infer_sector(headline)
    event = infer_event(headline)
    return {
        "headline": headline, "sentiment": label, "confidence": scores,
        "market_impact": impact, "impact_direction": impact_direction,
        "strength": strength, "company": company, "sector": sector, "event": event,
        "evidence_positive": evidence_pos, "evidence_negative": evidence_neg,
        "explanation": rule_explanation(label, impact, evidence_pos, evidence_neg, event),
        "investor_implications": rule_implications(label, event),
        "risk_factors": "Headline-only assessment; context, valuation, expectations, and market conditions may change the reaction.",
        "method": "Rule-based lexical analysis",
    }

def extract_company(headline: str) -> str:
    # Heuristic named-entity-like extraction; not a proper entity recognizer.
    candidates = ["Reliance Industries", "Tata Motors", "TCS", "Infosys", "HDFC Bank",
                  "ICICI Bank", "Wipro", "Adani Enterprises", "Adani Green", "Bharti Airtel",
                  "State Bank of India", "SBI", "Apple", "Microsoft", "Amazon", "Tesla",
                  "NVIDIA", "Google", "Meta", "Bajaj Finance", "Larsen & Toubro"]
    for c in candidates:
        if c.lower() in headline.lower():
            return c
    return "Not identified"

def infer_sector(headline: str) -> str:
    h = headline.lower()
    mapping = {
        "Energy / Renewables": ["renewable", "solar", "wind", "energy", "oil", "gas"],
        "Banking / Financial Services": ["bank", "lending", "loan", "credit", "finance"],
        "Technology": ["software", "technology", "tech", "ai", "cloud", "semiconductor", "chip"],
        "Automobiles": ["auto", "vehicle", "car", "ev", "electric vehicle"],
        "Healthcare / Pharma": ["pharma", "drug", "healthcare", "medicine", "clinical"],
        "Consumer / Retail": ["retail", "consumer", "sales", "store", "e-commerce"],
        "Telecommunications": ["telecom", "5g", "mobile", "network"],
        "Infrastructure / Industrials": ["infrastructure", "construction", "industrial", "project"],
    }
    for sector, words in mapping.items():
        if any(w in h for w in words):
            return sector
    return "General / Unclassified"

def infer_event(headline: str) -> str:
    h = headline.lower()
    event_map = [
        ("Earnings / Results", ["earnings", "results", "quarterly", "profit", "revenue"]),
        ("Regulatory / Legal", ["investigation", "regulatory", "lawsuit", "fraud", "court", "fine"]),
        ("Investment / Expansion", ["investment", "invests", "expansion", "expands", "project"]),
        ("Product / Technology", ["launch", "product", "technology", "ai", "innovation"]),
        ("Corporate Action", ["dividend", "buyback", "acquisition", "merger"]),
        ("Leadership", ["ceo", "resigns", "appointed", "leadership"]),
        ("Market / Macro", ["inflation", "interest rate", "recession", "economy"]),
    ]
    for event, words in event_map:
        if any(w in h for w in words):
            return event
    return "General News"

def rule_explanation(label, impact, pos, neg, event):
    if label == "Positive":
        cues = ", ".join(pos[:6]) if pos else "positive news cues"
        return f"The headline contains favorable signals ({cues}). Such information may improve expectations about earnings, growth, or business prospects. The inferred short-term tone is {impact.lower()}, but the actual reaction depends on expectations and valuation."
    if label == "Negative":
        cues = ", ".join(neg[:6]) if neg else "negative news cues"
        return f"The headline contains adverse signals ({cues}). Investors may reassess profitability, legal exposure, execution, or uncertainty. The inferred short-term tone is {impact.lower()}, although the news may already be priced in."
    return f"The headline appears largely factual or mixed, without a strong directional signal. Investors may wait for additional details about {event.lower()} before changing expectations."

def rule_implications(label, event):
    if label == "Positive":
        return "Potentially supportive of investor confidence and valuation expectations if the news is better than anticipated. Check whether the market had already priced in the event."
    if label == "Negative":
        return "Potentially negative for confidence or valuation if the news worsens expected cash flows or increases uncertainty. Assess materiality and possible remediation."
    return "Potentially limited immediate directional effect. Investors may focus on details, forward guidance, and subsequent disclosures."

def openai_analysis(headline: str, model: str, api_key: str) -> Dict:
    if OpenAI is None:
        raise RuntimeError("OpenAI package not installed. Install requirements or use rule-based mode.")
    client = OpenAI(api_key=api_key)
    prompt = f"""Analyze this financial news headline for an educational finance class.
Headline: {headline}
Return ONLY valid JSON with these fields:
sentiment (Positive, Neutral, Negative),
confidence (object with Positive, Neutral, Negative integer percentages totaling 100),
market_impact (Bullish, Bearish, Neutral / Mixed, Neutral / Unclear),
strength (Low, Moderate, High),
company (string or Not identified),
sector (string),
event (string),
explanation (2-4 sentences explaining why this news could influence investors),
investor_implications (1-3 sentences, conditional and balanced),
risk_factors (1-3 sentences explaining uncertainty, what headline omits, and that markets may have priced it in).
Do not offer buy/sell advice. Distinguish headline tone from actual price direction.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful financial-news sentiment analyst for education. Avoid certainty and investment advice. Output only JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    # Validate and normalize confidence values.
    conf = result.get("confidence", {})
    values = [max(0, int(conf.get(k, 0))) for k in ["Positive", "Neutral", "Negative"]]
    total = sum(values)
    if total <= 0:
        values = [33, 34, 33]
    else:
        values = [round(v * 100 / total) for v in values]
    values[1] += 100 - sum(values)
    result["confidence"] = dict(zip(["Positive", "Neutral", "Negative"], values))
    result["headline"] = headline
    result["method"] = f"LLM analysis ({model})"
    result.setdefault("company", extract_company(headline))
    result.setdefault("sector", infer_sector(headline))
    result.setdefault("event", infer_event(headline))
    result.setdefault("strength", "Moderate")
    result.setdefault("market_impact", "Neutral / Mixed")
    result.setdefault("explanation", "No explanation was returned.")
    result.setdefault("investor_implications", "Review company disclosures and broader market conditions.")
    result.setdefault("risk_factors", "Headlines provide incomplete information; sentiment is not a price forecast.")
    return result

def analyze(headline, mode, model, api_key):
    if mode == "OpenAI API" and api_key.strip():
        try:
            return openai_analysis(headline, model, api_key.strip())
        except Exception as e:
            st.warning(f"LLM analysis failed; using rule-based analysis instead. Details: {e}")
    return rule_analysis(headline)

def normalize_result(result):
    result["confidence"] = result.get("confidence", {"Positive": 33, "Neutral": 34, "Negative": 33})
    for k in ("Positive", "Neutral", "Negative"):
        result["confidence"].setdefault(k, 0)
    return result

def load_example(example: str) -> None:
    """Select a classroom example before Streamlit reruns the script."""
    st.session_state["selected_demo_headline"] = example
    st.session_state["single_headline"] = example


# ------------------------------ Sidebar ---------------------------------------
st.sidebar.title("⚙️ Analysis Settings")
mode = st.sidebar.radio(
    "Analysis engine",
    ["Rule-based (offline)", "OpenAI API"],
    help="Rule-based analysis works without an API key. OpenAI mode generates a richer explanation.",
)
api_key = st.sidebar.text_input(
    "OpenAI API key (optional)", type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
    help="Your key is used only for this session. Prefer setting OPENAI_API_KEY in your environment or Streamlit secrets.",
)
model = st.sidebar.selectbox("LLM model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Educational prototype. Headlines are simplified; outputs are not investment recommendations.")

# ------------------------------ Header ----------------------------------------
st.title("📰 Financial News Sentiment Analyzer")
st.markdown("**Generative AI + Finance | Classroom Demonstration**")
st.write("Explore how a financial news headline may shape investor sentiment and perceived market impact.")
st.info("The analyzer estimates the tone of a headline. It does **not** predict stock prices or recommend trades.")

tab_single, tab_batch, tab_demo, tab_about = st.tabs(["🔎 Single headline", "📚 Batch analysis", "🎓 Classroom examples", "ℹ️ About"])

# ------------------------------ Single ----------------------------------------
with tab_single:
    default_headline = "Reliance Industries reports strong quarterly earnings and announces major investment in renewable energy."
    headline = st.text_area("Enter a financial news headline", default_headline, height=95, key="single_headline")
    col1, col2 = st.columns([1, 4])
    with col1:
        run_single = st.button("Analyze headline", type="primary", use_container_width=True)
    with col2:
        st.caption("Tip: Compare a positive headline with a negative or mixed headline about the same company.")
    if run_single and headline.strip():
        result = normalize_result(analyze(headline.strip(), mode, model, api_key))
        st.session_state["last_result"] = result
        st.session_state["last_headline"] = headline.strip()

    result = st.session_state.get("last_result")
    if result:
        st.markdown("---")
        sentiment = result["sentiment"].upper()
        if sentiment == "POSITIVE":
            st.success(f"Sentiment: **{sentiment}**")
        elif sentiment == "NEGATIVE":
            st.error(f"Sentiment: **{sentiment}**")
        else:
            st.warning(f"Sentiment: **{sentiment}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Likely market tone", result.get("market_impact", "Unclear"))
        m2.metric("Sentiment strength", result.get("strength", "N/A"))
        m3.metric("Company", result.get("company", "Not identified"))
        m4.metric("Event category", result.get("event", "General News"))
        chart_df = pd.DataFrame({
            "Sentiment": list(result["confidence"].keys()),
            "Probability (%)": list(result["confidence"].values()),
        })
        fig = px.bar(chart_df, x="Sentiment", y="Probability (%)", color="Sentiment",
                     text="Probability (%)", range_y=[0, 100], title="Sentiment probability distribution")
        fig.update_layout(showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True, key="single_headline_sentiment_chart")
        st.markdown("### 🧠 Why might investors react?")
        st.write(result.get("explanation", ""))
        st.markdown("### 📌 Potential investor implications")
        st.write(result.get("investor_implications", ""))
        st.markdown("### ⚠️ Risks and limitations")
        st.write(result.get("risk_factors", ""))
        st.markdown("### 🏷️ Context clues")
        c1, c2, c3 = st.columns(3)
        c1.write(f"**Sector:** {result.get('sector', 'N/A')}")
        c2.write(f"**Event:** {result.get('event', 'N/A')}")
        c3.write(f"**Method:** {result.get('method', 'N/A')}")
        with st.expander("Show lexical evidence (rule-based mode)"):
            positive_cues = result.get("evidence_positive", [])
            negative_cues = result.get("evidence_negative", [])
            st.text("Positive cues: " + (", ".join(positive_cues) if positive_cues else "None detected"))
            st.text("Negative cues: " + (", ".join(negative_cues) if negative_cues else "None detected"))
        st.download_button(
            "Download this analysis as JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="headline_sentiment_analysis.json",
            mime="application/json",
        )

# ------------------------------ Batch -----------------------------------------
with tab_batch:
    st.markdown("### Analyze multiple headlines")
    st.write("Enter one headline per line to compare sentiment across a set of news items.")
    batch_text = st.text_area("Headlines", height=220, placeholder="Headline 1\nHeadline 2\nHeadline 3", key="batch_text")
    batch_run = st.button("Analyze batch", key="batch_run")
    if batch_run and batch_text.strip():
        lines = [line.strip() for line in batch_text.splitlines() if line.strip()]
        batch_results = [normalize_result(analyze(line, mode, model, api_key)) for line in lines]
        rows = []
        for r in batch_results:
            rows.append({
                "Headline": r["headline"],
                "Sentiment": r["sentiment"],
                "Market impact": r.get("market_impact"),
                "Confidence (%)": max(r["confidence"].values()),
                "Positive (%)": r["confidence"]["Positive"],
                "Neutral (%)": r["confidence"]["Neutral"],
                "Negative (%)": r["confidence"]["Negative"],
                "Company": r.get("company"),
                "Sector": r.get("sector"),
                "Event": r.get("event"),
                "Explanation": r.get("explanation"),
                "Method": r.get("method"),
            })
        df = pd.DataFrame(rows)
        st.session_state["batch_df"] = df
        st.session_state["batch_results"] = batch_results
    if "batch_df" in st.session_state:
        df = st.session_state["batch_df"]
        st.dataframe(df, use_container_width=True, hide_index=True)
        counts = df["Sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Headlines"]
        st.plotly_chart(px.pie(counts, names="Sentiment", values="Headlines",
                               title="Sentiment composition of batch", hole=.35), use_container_width=True,
                               key="batch_sentiment_composition_chart")
        st.download_button("Download batch results (CSV)", df.to_csv(index=False).encode("utf-8"),
                          file_name="news_sentiment_batch.csv", mime="text/csv")
        st.download_button("Download full batch analysis (JSON)",
                          json.dumps(st.session_state["batch_results"], indent=2, ensure_ascii=False),
                          file_name="news_sentiment_batch.json", mime="application/json")

# ------------------------------ Demo examples ---------------------------------
with tab_demo:
    st.markdown("### Try these classroom examples")
    st.write(
        "Click **Use this headline** to select an example. The selected headline "
        "will appear below, where you can analyze it immediately. It is also copied "
        "to the Single headline tab."
    )

    examples = [
        (
            "Positive / Earnings",
            "Reliance Industries reports strong quarterly earnings and announces major investment in renewable energy.",
        ),
        (
            "Negative / Regulatory",
            "Company faces regulatory investigation after disappointing quarterly results.",
        ),
        (
            "Positive / Product",
            "Technology firm launches innovative product and raises full-year revenue guidance.",
        ),
        (
            "Negative / Risk",
            "Major manufacturer recalls products after safety concerns and faces a potential fine.",
        ),
        (
            "Neutral / Corporate",
            "Company announces appointment of a new chief operating officer.",
        ),
        (
            "Mixed / Results",
            "Revenue rises sharply, but profit margins decline and management warns of higher costs.",
        ),
    ]

    for category, example in examples:
        with st.expander(category):
            st.write(example)
            st.button(
                "Use this headline",
                key=f"use_{category}",
                on_click=load_example,
                args=(example,),
                use_container_width=True,
            )

    selected_example = st.session_state.get("selected_demo_headline")
    if selected_example:
        st.markdown("---")
        st.markdown("### Selected classroom headline")
        st.info(selected_example)
        st.success("Headline selected successfully.")
        st.caption(
            "The selected headline is also copied into the Single headline input. "
            "You can analyze it here or switch to the Single headline tab."
        )
        if st.button("Analyze selected example", type="primary", key="analyze_demo_example"):
            demo_result = normalize_result(analyze(selected_example, mode, model, api_key))
            st.session_state["last_result"] = demo_result
            st.session_state["last_headline"] = selected_example

            sentiment = demo_result.get("sentiment", "Neutral").upper()
            if sentiment == "POSITIVE":
                st.success(f"Sentiment: **{sentiment}**")
            elif sentiment == "NEGATIVE":
                st.error(f"Sentiment: **{sentiment}**")
            else:
                st.warning(f"Sentiment: **{sentiment}**")

            d1, d2, d3 = st.columns(3)
            d1.metric("Likely market tone", demo_result.get("market_impact", "Unclear"))
            d2.metric("Sentiment strength", demo_result.get("strength", "N/A"))
            d3.metric("Event category", demo_result.get("event", "General News"))

            demo_conf = demo_result.get("confidence", {})
            demo_chart = pd.DataFrame({
                "Sentiment": list(demo_conf.keys()),
                "Probability (%)": list(demo_conf.values()),
            })
            demo_fig = px.bar(
                demo_chart, x="Sentiment", y="Probability (%)",
                color="Sentiment", text="Probability (%)",
                range_y=[0, 100], title="Sentiment probability distribution",
            )
            demo_fig.update_layout(showlegend=False)
            demo_fig.update_traces(textposition="outside")
            st.plotly_chart(demo_fig, use_container_width=True, key="classroom_demo_sentiment_chart")

            st.markdown("**Why might investors react?**")
            st.write(demo_result.get("explanation", ""))
            st.markdown("**Potential investor implications**")
            st.write(demo_result.get("investor_implications", ""))
            st.markdown("**Risks and limitations**")
            st.write(demo_result.get("risk_factors", ""))


# ------------------------------ About -----------------------------------------
with tab_about:
    st.markdown("""
### How the app works

**Rule-based mode**
- Uses a finance-oriented word and phrase dictionary.
- Identifies directional cues, event type, and rough sector.
- Produces approximate confidence scores for teaching purposes.
- Works without internet access or an API key.

**OpenAI API mode**
- Sends the headline to the selected language model.
- Requests sentiment probabilities, market tone, and a contextual explanation.
- Uses JSON-format responses to make the output structured.
- Requires a valid API key and network access.

### Suggested classroom activity
1. Analyze a positive headline.
2. Analyze a negative headline.
3. Analyze a mixed headline.
4. Compare rule-based and LLM results.
5. Discuss whether sentiment equals price movement.
6. Ask what missing information could reverse the interpretation.

### Important limitations
- A headline is not the complete news story.
- Sentiment is not the same as a forecast of share-price movement.
- Markets react to surprises relative to expectations, not only to whether information sounds positive or negative.
- Results may miss sarcasm, context, company-specific details, and sector nuance.
- Rule-based confidence scores are illustrative and are not statistically calibrated.
- LLM outputs can be incorrect or overconfident.
- The app is for educational use, not financial advice.
""")

st.markdown("---")
st.caption(f"Financial News Sentiment Analyzer | Educational prototype | Session date: {datetime.now().strftime('%d %b %Y')}")
