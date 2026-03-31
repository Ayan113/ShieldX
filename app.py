import os
import json
import re
import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


# Load GROQ_API_KEY from .env at the very top (required for Groq calls)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="ShieldX", page_icon="🛡️", layout="wide")


# Page header shown on every load
st.markdown("## 🛡️ ShieldX")
st.markdown("AI-Powered Crypto Fraud Detection & Compliance Platform")
st.divider()


# Initialize session state at app start
if "transaction_history" not in st.session_state:
    st.session_state["transaction_history"] = []
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "pending_rag_question" not in st.session_state:
    st.session_state["pending_rag_question"] = None


MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_groq_llm() -> ChatGroq:
    # A single place to configure the LLM.
    # Passing api_key explicitly makes behavior predictable even if env vars change.
    return ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)


def score_to_colors(fraud_score: int):
    if fraud_score < 30:
        return {"metric_bg": "#2E7D32", "badge_bg": "#1B5E20", "font": "white", "label": "green"}
    if fraud_score <= 60:
        return {"metric_bg": "#F9A825", "badge_bg": "#F57C00", "font": "black", "label": "yellow"}
    return {"metric_bg": "#C62828", "badge_bg": "#B71C1C", "font": "white", "label": "red"}


def risk_level_to_badge_color(risk_level: str):
    mapping = {
        "LOW": ("#2E7D32", "white"),
        "MEDIUM": ("#F9A825", "black"),
        "HIGH": ("#EF6C00", "white"),
        "CRITICAL": ("#C62828", "white"),
    }
    return mapping.get(risk_level, ("#9E9E9E", "white"))


def recommendation_to_banner(rec: str):
    mapping = {
        "APPROVE": ("#2E7D32", "white"),
        "REVIEW": ("#F9A825", "black"),
        "BLOCK": ("#C62828", "white"),
    }
    return mapping.get(rec, ("#9E9E9E", "white"))


def extract_first_json_object(text: str) -> str:
    # Attempts to extract the first JSON object substring from the LLM output.
    # This reduces failures when the model adds stray whitespace or text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in the model output.")
    return text[start : end + 1]


def normalize_fraud_response(obj):
    expected = {
        "fraud_score": int,
        "risk_level": str,
        "fraud_indicators": list,
        "legitimate_indicators": list,
        "recommendation": str,
        "explanation": str,
        "similar_fraud_patterns": list,
    }
    missing = [k for k in expected.keys() if k not in obj]
    if missing:
        raise ValueError(f"Missing required fields in JSON: {missing}")

    fraud_score = obj["fraud_score"]
    if isinstance(fraud_score, float) and fraud_score.is_integer():
        fraud_score = int(fraud_score)
    fraud_score = int(fraud_score)
    fraud_score = max(0, min(100, fraud_score))

    risk_level = str(obj["risk_level"]).upper()
    recommendation = str(obj["recommendation"]).upper()
    allowed_risk = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    allowed_rec = {"APPROVE", "REVIEW", "BLOCK"}
    if risk_level not in allowed_risk:
        raise ValueError(f"Invalid risk_level: {risk_level}. Must be one of {sorted(allowed_risk)}.")
    if recommendation not in allowed_rec:
        raise ValueError(
            f"Invalid recommendation: {recommendation}. Must be one of {sorted(allowed_rec)}."
        )

    fraud_indicators = obj["fraud_indicators"]
    legitimate_indicators = obj["legitimate_indicators"]
    similar_fraud_patterns = obj["similar_fraud_patterns"]

    if not isinstance(fraud_indicators, list) or not all(isinstance(x, str) for x in fraud_indicators):
        raise ValueError("fraud_indicators must be a list of strings.")
    if not isinstance(legitimate_indicators, list) or not all(
        isinstance(x, str) for x in legitimate_indicators
    ):
        raise ValueError("legitimate_indicators must be a list of strings.")
    if not isinstance(similar_fraud_patterns, list) or not all(
        isinstance(x, str) for x in similar_fraud_patterns
    ):
        raise ValueError("similar_fraud_patterns must be a list of strings.")

    explanation = obj["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be a non-empty string.")

    return {
        "fraud_score": fraud_score,
        "risk_level": risk_level,
        "fraud_indicators": fraud_indicators,
        "legitimate_indicators": legitimate_indicators,
        "recommendation": recommendation,
        "explanation": explanation.strip(),
        "similar_fraud_patterns": similar_fraud_patterns,
    }


def analyze_transaction_with_llm(transaction_fields: dict):
    if not GROQ_API_KEY:
        st.error("Missing `GROQ_API_KEY`. Add it to the `.env` file in the project root.")
        return None

    # Build the prompt with all fields.
    prompt = (
        "You are a crypto fraud detection expert and compliance officer.\n"
        "Analyze the following crypto transaction and return ONLY a valid JSON object (no markdown, "
        "no code blocks, no extra text).\n\n"
        "Return JSON with these exact fields:\n"
        "- fraud_score: integer 0-100\n"
        '- risk_level: one of "LOW","MEDIUM","HIGH","CRITICAL"\n'
        "- fraud_indicators: list of strings describing suspicious patterns found\n"
        "- legitimate_indicators: list of strings describing normal patterns found\n"
        '- recommendation: one of "APPROVE","REVIEW","BLOCK"\n'
        "- explanation: 2-3 sentence plain English explanation\n"
        "- similar_fraud_patterns: list of known fraud pattern names this resembles\n\n"
        "Transaction details:\n"
        f"transaction_id: {transaction_fields.get('transaction_id')}\n"
        f"sender_wallet: {transaction_fields.get('sender_wallet')}\n"
        f"receiver_wallet: {transaction_fields.get('receiver_wallet')}\n"
        f"amount_usd: {transaction_fields.get('amount_usd')}\n"
        f"crypto_type: {transaction_fields.get('crypto_type')}\n"
        f"timestamp: {transaction_fields.get('timestamp')}\n"
        f"sender_account_age_days: {transaction_fields.get('sender_account_age_days')}\n"
        f"transactions_last_24h: {transaction_fields.get('transactions_last_24h')}\n"
        f"is_cross_border: {transaction_fields.get('is_cross_border')}\n"
        f"transaction_fee_usd: {transaction_fields.get('transaction_fee_usd')}\n"
    )

    try:
        llm = get_groq_llm()
        resp = llm.invoke(
            [
                SystemMessage(
                    content="Return ONLY valid JSON. Do not include markdown or any extra text."
                ),
                HumanMessage(content=prompt),
            ]
        )
        raw = resp.content if hasattr(resp, "content") else str(resp)

        try:
            extracted = extract_first_json_object(raw)
            obj = json.loads(extracted)
        except Exception as e:
            st.error(f"Failed to parse JSON from Groq. Error: {e}")
            st.error("Model output was (truncated):")
            st.code(raw[:1200])
            return None

        try:
            normalized = normalize_fraud_response(obj)
        except Exception as e:
            st.error(f"Groq returned JSON but it did not match the required schema. Error: {e}")
            st.error("Parsed JSON was (truncated):")
            st.code(json.dumps(obj, ensure_ascii=False)[:1200])
            return None

        return normalized
    except Exception as e:
        st.error(f"Groq analysis call failed: {e}")
        return None


def render_analysis_result(analysis: dict):
    # Fraud score metric with colored background
    score = int(analysis["fraud_score"])
    colors = score_to_colors(score)
    st.markdown(
        f"""
        <div style="padding:14px;border-radius:14px;background-color:{colors["metric_bg"]};color:{colors["font"]};">
          <div style="font-size:14px;opacity:0.95;font-weight:600;">Fraud Score</div>
          <div style="font-size:44px;line-height:1.1;font-weight:800;">{score}</div>
          <div style="font-size:14px;opacity:0.95;font-weight:600;">0-100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk level badge
    risk_bg, risk_font = risk_level_to_badge_color(analysis["risk_level"])
    st.markdown(
        f"""
        <div style="margin-top:12px;padding:10px 14px;border-radius:999px;background-color:{risk_bg};color:{risk_font};display:inline-block;font-weight:800;">
          Risk Level: {analysis["risk_level"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Recommendation banner
    rec_bg, rec_font = recommendation_to_banner(analysis["recommendation"])
    st.markdown(
        f"""
        <div style="margin-top:12px;padding:12px 16px;border-radius:12px;background-color:{rec_bg};color:{rec_font};font-weight:900;">
          Recommendation: {analysis["recommendation"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Indicators")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Suspicious / Fraud Indicators")
        if analysis["fraud_indicators"]:
            st.markdown(
                "<ul style='margin:0;padding-left:18px;'>"
                + "".join([f"<li style='color:#C62828;font-weight:600;'>{x}</li>" for x in analysis["fraud_indicators"]])
                + "</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.write("No explicit fraud indicators were returned.")

    with right:
        st.markdown("#### Legitimate / Normal Indicators")
        if analysis["legitimate_indicators"]:
            st.markdown(
                "<ul style='margin:0;padding-left:18px;'>"
                + "".join(
                    [
                        f"<li style='color:#2E7D32;font-weight:600;'>{x}</li>"
                        for x in analysis["legitimate_indicators"]
                    ]
                )
                + "</ul>",
                unsafe_allow_html=True,
            )
        else:
            st.write("No explicit legitimate indicators were returned.")

    st.markdown("### Explanation")
    st.write(analysis["explanation"])

    st.markdown("### Similar Fraud Patterns")
    if analysis["similar_fraud_patterns"]:
        st.markdown(
            "".join(
                [
                    f"<span style='display:inline-block;margin:6px 6px 0 0;padding:6px 10px;border-radius:999px;background-color:#F3F4F6;color:#111827;font-weight:700;border:1px solid #E5E7EB;'>{p}</span>"
                    for p in analysis["similar_fraud_patterns"]
                ]
            ),
            unsafe_allow_html=True,
        )
    else:
        st.write("No similar fraud patterns were returned.")


def generate_wallet(prefix="0x", length=40):
    chars = "0123456789abcdef"
    return prefix + "".join(random.choice(chars) for _ in range(length))


def generate_sample_transactions_csv_bytes() -> bytes:
    # Generates realistic-looking fake transactions for the sample CSV only.
    random.seed(42)
    now = datetime.now()
    rows = []

    crypto_types = ["Bitcoin", "Ethereum", "USDT", "Solana", "XRP", "Other"]
    # Some known-ish "threshold" behavior around 10,000 to make sample include structuring.
    threshold = 10000

    fraudulent_senders = [
        ("0xabc123abc123abc123abc123abc123abc123abcd", "0x9f1e2d3c4b5a6f7e8d9c0b1a2c3d4e5f6a7b8c9d"),
        ("0x1111222233334444555566667777888899990000", "0x22223333444455556666777788889999aaaabbbb"),
        ("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "0xcafebabecafebabecafebabecafebabecafebabe"),
    ]

    for i in range(10):
        tx_id = f"tx_{i+1:04d}"

        crypto_type = random.choice(crypto_types)
        is_cross_border = random.random() < 0.35

        sender_wallet = generate_wallet()
        receiver_wallet = generate_wallet()
        amount = round(random.uniform(50, 45000), 2)
        if random.random() < 0.25:
            # Create suspicious "just under threshold" amounts for structuring-like behavior
            amount = round(threshold - random.uniform(10, 650), 2)

        sender_age_days = random.randint(1, 2500)
        transactions_last_24h = random.randint(0, 25)

        transaction_fee_usd = round(random.uniform(0.5, 55.0), 2)
        ts = now - timedelta(hours=random.randint(1, 720))
        timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")

        # Force a few rows to be clearly fraudulent-looking.
        if i in (2, 5, 8):
            sender_wallet, receiver_wallet = fraudulent_senders[(i // 3) % len(fraudulent_senders)]
            sender_age_days = random.randint(0, 45)
            transactions_last_24h = random.randint(18, 60)
            amount = round(threshold - random.uniform(1, 75), 2)
            is_cross_border = True
            transaction_fee_usd = round(random.uniform(1.0, 20.0), 2)

        if i in (7,):
            # Pump/dump-ish: large fast movement (sample heuristic)
            sender_age_days = random.randint(0, 90)
            transactions_last_24h = random.randint(10, 40)
            amount = round(random.uniform(15000, 80000), 2)
            is_cross_border = random.random() < 0.7

        rows.append(
            {
                "transaction_id": tx_id,
                "sender_wallet": sender_wallet,
                "receiver_wallet": receiver_wallet,
                "amount_usd": amount,
                "crypto_type": crypto_type,
                "timestamp": timestamp,
                "sender_account_age_days": sender_age_days,
                "transactions_last_24h": transactions_last_24h,
                "is_cross_border": int(is_cross_border),
                "transaction_fee_usd": transaction_fee_usd,
            }
        )

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def coerce_is_cross_border(val):
    if isinstance(val, bool):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return False


def coerce_number(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def parse_dataframe_row(row: pd.Series) -> dict:
    # Convert CSV row to the field names expected by the prompt.
    timestamp_val = row.get("timestamp")
    try:
        ts_parsed = pd.to_datetime(timestamp_val, errors="coerce")
        timestamp_str = (
            ts_parsed.strftime("%Y-%m-%d %H:%M:%S") if ts_parsed is not pd.NaT else str(timestamp_val)
        )
    except Exception:
        timestamp_str = str(timestamp_val)

    return {
        "transaction_id": str(row.get("transaction_id")),
        "sender_wallet": str(row.get("sender_wallet")),
        "receiver_wallet": str(row.get("receiver_wallet")),
        "amount_usd": float(row.get("amount_usd")),
        "crypto_type": str(row.get("crypto_type")),
        "timestamp": timestamp_str,
        "sender_account_age_days": int(float(row.get("sender_account_age_days"))),
        "transactions_last_24h": int(float(row.get("transactions_last_24h"))),
        "is_cross_border": bool(coerce_is_cross_border(row.get("is_cross_border"))),
        "transaction_fee_usd": float(row.get("transaction_fee_usd")),
    }


# 4 tabs layout (main app)
tabs = st.tabs(
    [
        "Single Transaction Analyzer",
        "Batch CSV Analyzer",
        "Fraud Pattern Dashboard",
        "Fraud Rules RAG Chatbot",
    ]
)


with tabs[0]:
    # -------------------- Tab 1: Single Transaction Analyzer --------------------
    st.subheader("Single Transaction Analyzer")

    with st.form("single_transaction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            transaction_id = st.text_input("Transaction ID", placeholder="tx_0001")
            sender_wallet = st.text_input(
                "Sender Wallet Address", placeholder="0x1a2b3c..."
            )
            receiver_wallet = st.text_input(
                "Receiver Wallet Address", placeholder="0x9z8y7x..."
            )
            amount_usd = st.number_input("Amount in USD", min_value=0.0, value=0.0, step=0.01)
            crypto_type = st.selectbox(
                "Cryptocurrency type",
                ["Bitcoin", "Ethereum", "USDT", "Solana", "XRP", "Other"],
                index=1,
            )
        with col2:
            transaction_timestamp = st.datetime_input(
                "Transaction timestamp",
                value=datetime.now(),
            )
            sender_account_age_days = st.number_input(
                "Sender account age in days",
                min_value=0,
                value=0,
                step=1,
            )
            transactions_last_24h = st.number_input(
                "Number of transactions sender made in last 24 hours",
                min_value=0,
                value=0,
                step=1,
            )
            is_cross_border = st.checkbox("Is this a cross-border transaction?")
            transaction_fee_usd = st.number_input(
                "Transaction fee in USD",
                min_value=0.0,
                value=0.0,
                step=0.01,
            )

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted:
        fields = {
            "transaction_id": transaction_id,
            "sender_wallet": sender_wallet,
            "receiver_wallet": receiver_wallet,
            "amount_usd": amount_usd,
            "crypto_type": crypto_type,
            "timestamp": transaction_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_account_age_days": int(sender_account_age_days),
            "transactions_last_24h": int(transactions_last_24h),
            "is_cross_border": bool(is_cross_border),
            "transaction_fee_usd": transaction_fee_usd,
        }

        with st.spinner("Analyzing transaction with Groq..."):
            analysis = analyze_transaction_with_llm(fields)

        if analysis:
            st.markdown("## Analysis Result")
            render_analysis_result(analysis)

            st.session_state["transaction_history"].append(
                {
                    "transaction": fields,
                    "analysis": analysis,
                    "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

with tabs[1]:
    # -------------------- Tab 2: Batch CSV Analyzer --------------------
    st.subheader("Batch CSV Analyzer")

    st.markdown("### Sample CSV Format")
    st.code(
        "transaction_id, sender_wallet, receiver_wallet, amount_usd, crypto_type, timestamp, sender_account_age_days, transactions_last_24h, is_cross_border, transaction_fee_usd"
    )

    sample_bytes = generate_sample_transactions_csv_bytes()
    st.download_button(
        label="Download Sample CSV",
        data=sample_bytes,
        file_name="shieldx_sample_transactions.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.markdown("### Preview (first 5 rows)")
            st.dataframe(df_in.head(5), use_container_width=True)

            expected_columns = [
                "transaction_id",
                "sender_wallet",
                "receiver_wallet",
                "amount_usd",
                "crypto_type",
                "timestamp",
                "sender_account_age_days",
                "transactions_last_24h",
                "is_cross_border",
                "transaction_fee_usd",
            ]
            missing_cols = [c for c in expected_columns if c not in df_in.columns]
            if missing_cols:
                st.error(f"CSV is missing required columns: {missing_cols}")
            else:
                analyze_all = st.button("Analyze All Transactions")
                if analyze_all:
                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    total = len(df_in)
                    for idx, row in df_in.iterrows():
                        status_text = f"Analyzing transaction {idx + 1}/{total}..."
                        status.write(status_text)
                        progress.progress((idx + 1) / max(total, 1))

                        fields = parse_dataframe_row(row)
                        with st.spinner(f"Groq analyzing {fields['transaction_id']}..."):
                            analysis = analyze_transaction_with_llm(fields)

                        if analysis:
                            results.append({"transaction": fields, "analysis": analysis})
                        else:
                            results.append({"transaction": fields, "analysis": None})

                    st.session_state["batch_results"] = results
                    progress.progress(1.0)

                    valid = [r for r in results if r.get("analysis") is not None]
                    if valid:
                        summary_rows = [
                            {
                                "transaction_id": r["transaction"]["transaction_id"],
                                "amount_usd": r["transaction"]["amount_usd"],
                                "risk_level": r["analysis"]["risk_level"],
                                "fraud_score": r["analysis"]["fraud_score"],
                                "recommendation": r["analysis"]["recommendation"],
                            }
                            for r in valid
                        ]
                        summary_df = pd.DataFrame(summary_rows)

                        st.markdown("### Analysis Summary")
                        st.dataframe(summary_df, use_container_width=True)

                        total_flagged = int(
                            (summary_df["risk_level"].isin(["HIGH", "CRITICAL"])).sum()
                        )
                        total_blocked = int((summary_df["recommendation"] == "BLOCK").sum())
                        total_approved = int((summary_df["recommendation"] == "APPROVE").sum())

                        highest_idx = summary_df["fraud_score"].astype(float).idxmax()
                        highest = summary_df.loc[highest_idx].to_dict()

                        st.markdown("### Aggregate Stats")
                        a, b, c, d = st.columns(4, gap="small")
                        a.metric("Total Flagged", total_flagged)
                        b.metric("Total Blocked", total_blocked)
                        c.metric("Total Approved", total_approved)
                        d.metric(
                            "Highest Risk Transaction",
                            f'{highest.get("transaction_id")} - {highest.get("fraud_score")} ({highest.get("risk_level")}, {highest.get("recommendation")})',
                        )

                    status.write("Batch analysis complete.")

        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    else:
        st.info("Upload a CSV to analyze multiple transactions.")

with tabs[2]:
    # -------------------- Tab 3: Fraud Pattern Dashboard --------------------
    st.subheader("Fraud Pattern Dashboard")

    transaction_items = st.session_state.get("transaction_history", [])
    batch_items = st.session_state.get("batch_results", [])
    combined = []

    for item in transaction_items:
        if item and item.get("analysis"):
            combined.append(item)
    for item in batch_items:
        if item and item.get("analysis"):
            combined.append(item)

    if not combined:
        st.info("Analyze some transactions first to see patterns here.")
    else:
        rows = []
        for it in combined:
            tx = it["transaction"]
            analysis = it["analysis"]
            rows.append(
                {
                    "transaction_id": tx.get("transaction_id"),
                    "amount_usd": float(tx.get("amount_usd", 0.0)),
                    "fraud_score": int(analysis.get("fraud_score")),
                    "risk_level": analysis.get("risk_level"),
                    "recommendation": analysis.get("recommendation"),
                }
            )
        df = pd.DataFrame(rows)

        risk_colors = {"LOW": "#2E7D32", "MEDIUM": "#F9A825", "HIGH": "#EF6C00", "CRITICAL": "#C62828"}
        rec_colors = {"APPROVE": "#2E7D32", "REVIEW": "#F9A825", "BLOCK": "#C62828"}

        # 2x2 grid charts
        st.markdown("### Key Charts")
        c1, c2 = st.columns(2, gap="large")
        c3, c4 = st.columns(2, gap="large")

        with c1:
            fig1 = px.pie(
                df,
                names="risk_level",
                hole=0.35,
                color="risk_level",
                color_discrete_map=risk_colors,
            )
            fig1.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=df["fraud_score"], nbinsx=20, marker_color="#7B1FA2"))
            fig2.add_vline(x=60, line_width=3, line_dash="dash", line_color="red")
            fig2.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Fraud Score",
                yaxis_title="Count",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with c3:
            fig3 = px.scatter(
                df,
                x="amount_usd",
                y="fraud_score",
                color="risk_level",
                color_discrete_map=risk_colors,
                hover_data=["transaction_id", "recommendation"],
            )
            fig3.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Amount (USD)",
                yaxis_title="Fraud Score",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            rec_counts = df["recommendation"].value_counts().reindex(["APPROVE", "REVIEW", "BLOCK"]).fillna(0)
            fig4 = go.Figure(
                go.Bar(
                    x=rec_counts.values,
                    y=rec_counts.index,
                    orientation="h",
                    marker_color=[rec_colors.get(k, "#9E9E9E") for k in rec_counts.index],
                )
            )
            fig4.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Count",
                yaxis_title="Recommendation",
                yaxis=dict(categoryorder="array", categoryarray=["APPROVE", "REVIEW", "BLOCK"]),
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### Flagged Transactions (HIGH / CRITICAL)")
        flagged_df = df[df["risk_level"].isin(["HIGH", "CRITICAL"])].copy()
        if flagged_df.empty:
            st.write("No HIGH/CRITICAL transactions found in your current session.")
        else:
            # A compact set of columns for review
            flagged_df = flagged_df[
                ["transaction_id", "amount_usd", "risk_level", "fraud_score", "recommendation"]
            ].sort_values(by="fraud_score", ascending=False)
            st.dataframe(flagged_df, use_container_width=True)

with tabs[3]:
    # -------------------- Tab 4: Fraud Rules RAG Chatbot --------------------
    st.subheader("Fraud Rules RAG Chatbot")

    @st.cache_resource
    def get_vector_store():
        # Embeds and stores the fixed knowledge chunks into ChromaDB on first load.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(base_dir, ".chroma_shieldx")

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        knowledge_chunks = [
            "Structuring or smurfing is a common crypto fraud pattern where large amounts are broken into many small transactions just below reporting thresholds to avoid detection. Typical threshold in the US is $10,000. Red flags: many transactions just under $10,000 from the same wallet within 24 hours.",
            "Pump and dump schemes in crypto involve artificially inflating a token's price through coordinated buying and misleading promotions, then selling at the peak. Indicators: sudden volume spike, many new wallets buying the same token, social media mentions spiking, price increase over 50% in under an hour.",
            "Rug pull fraud occurs when developers of a new crypto project suddenly withdraw all liquidity and abandon the project. Signs: anonymous team, no code audit, liquidity locked for very short period, token launched less than 30 days ago, trading volume drops suddenly to near zero.",
            "Wash trading involves simultaneously buying and selling the same crypto asset to generate artificial trading volume. Detection: same wallet appearing as both sender and receiver in related transactions, identical amounts traded back and forth, trades happening at regular exact intervals.",
            "Blockchain analytics red flags for money laundering: transactions going through multiple wallets in rapid succession, use of privacy coins like Monero to obscure trail, mixing services or tumblers detected in transaction path, funds originating from darknet market wallets.",
            "FATF guidelines for crypto: exchanges must implement KYC for transactions over $1,000, report suspicious transactions, maintain records for 5 years. The travel rule requires sharing sender and receiver information for transfers over $3,000.",
            "Phishing and account takeover in crypto: sudden transaction from a wallet that has never transacted before, transaction at unusual hours, transaction amount much larger than historical average, receiver wallet created less than 24 hours ago.",
            "High risk jurisdictions for crypto transactions: transactions involving wallets linked to sanctioned countries should be automatically flagged. Cross-border transactions with high-risk jurisdictions require enhanced due diligence.",
            "Velocity checks in crypto fraud detection: more than 10 transactions in 1 hour from the same wallet is suspicious, more than 3 failed transactions followed by a successful one indicates credential stuffing, rapid movement of funds to multiple wallets within minutes indicates layering.",
            "Crypto exchange compliance: exchanges must file Suspicious Activity Reports with FinCEN for transactions over $5,000 that appear suspicious. Customer due diligence must include identity verification and source of funds verification for large transactions.",
        ]

        metadatas = [{"chunk_id": i + 1} for i in range(len(knowledge_chunks))]
        texts = knowledge_chunks

        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name="shieldx_fraud_rules",
            persist_directory=persist_dir,
        )
        vectordb.persist()
        return vectordb

    # Build (and cache) the knowledge base on first load.
    with st.spinner("Preparing ShieldX knowledge base..."):
        _vectordb = get_vector_store()

    def rag_ask(question: str):
        # Runs the RAG flow: embed question, retrieve top 3 chunks, then call Groq.
        try:
            vectordb = _vectordb
            retrieved = vectordb.similarity_search(question, k=3)
        except Exception as e:
            st.error(f"Knowledge retrieval failed: {e}")
            return None, []

        context_chunks = []
        sources = []
        for doc in retrieved:
            cid = doc.metadata.get("chunk_id", "unknown")
            chunk_text = doc.page_content
            sources.append(f"Chunk {cid}: {chunk_text}")
            context_chunks.append(f"Chunk {cid}: {chunk_text}")

        context = "\n\n".join(context_chunks)
        rag_system_prompt = (
            "You are a crypto fraud detection expert and compliance officer. Answer questions about crypto fraud "
            "patterns, detection methods, and regulatory requirements using ONLY the context provided. Be specific and "
            "professional. If the answer is not in the context say: This specific information is not in my knowledge "
            "base, please consult FATF guidelines or FinCEN regulations directly."
        )

        user_prompt = (
            "Context:\n"
            f"{context}\n\n"
            f"Question: {question}\n"
        )

        if not GROQ_API_KEY:
            st.error("Missing `GROQ_API_KEY`. Add it to the `.env` file in the project root.")
            return None, sources

        try:
            llm = get_groq_llm()
            resp = llm.invoke(
                [
                    SystemMessage(content=rag_system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            answer_text = resp.content if hasattr(resp, "content") else str(resp)
            return answer_text, sources
        except Exception as e:
            st.error(f"Groq RAG call failed: {e}")
            return None, sources

    # If a suggested question was clicked in a previous rerun, process it first.
    if st.session_state.get("pending_rag_question"):
        question = st.session_state["pending_rag_question"]
        st.session_state["pending_rag_question"] = None
        with st.spinner("Retrieving knowledge and generating an answer..."):
            answer, sources = rag_ask(question)

        if answer is not None:
            st.session_state["rag_messages"].append({"role": "user", "content": question})
            st.session_state["rag_messages"].append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

    # Render chat history
    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("role") == "assistant" and msg.get("sources"):
                with st.expander("Knowledge Sources Used"):
                    st.write("\n".join(msg["sources"]))

    st.markdown("---")

    # Suggested clickable questions
    suggested_questions = [
        "What is structuring fraud?",
        "What are FATF guidelines?",
        "How do I detect wash trading?",
        "What are velocity check rules?",
    ]
    qcols = st.columns(2, gap="small")
    for i, q in enumerate(suggested_questions):
        with qcols[i % 2]:
            if st.button(q, key=f"suggest_{i}"):
                st.session_state["pending_rag_question"] = q
                st.rerun()

    # Chat input
    user_question = st.chat_input("Ask about fraud patterns, detection methods, or compliance requirements")
    if user_question:
        st.session_state["pending_rag_question"] = user_question
        st.rerun()

