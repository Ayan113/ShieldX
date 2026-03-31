import csv
import io
import json
import os
import random
import re
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from flask import Flask, Response, redirect, render_template, request, session, url_for
from groq import Groq
from plotly.utils import PlotlyJSONEncoder


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "shieldx-dev-secret")
MODEL_NAME = "llama-3.3-70b-versatile"
CRYPTO_TYPES = ["Bitcoin", "Ethereum", "USDT", "Solana", "XRP", "Other"]
KNOWLEDGE_CHUNKS = [
    "Structuring or smurfing is a common crypto fraud pattern where large amounts are broken into many small transactions just below reporting thresholds to avoid detection. Typical threshold in the US is $10,000. Red flags: many transactions just under $10,000 from the same wallet within 24 hours.",
    "Pump and dump schemes in crypto involve artificially inflating a token's price through coordinated buying and misleading promotions, then selling at the peak. Indicators: sudden volume spike, many new wallets buying the same token, social media mentions spiking, price increase over 50 percent in under an hour.",
    "Rug pull fraud occurs when developers of a new crypto project suddenly withdraw all liquidity and abandon the project. Signs: anonymous team, no code audit, liquidity locked for very short period, token launched less than 30 days ago, trading volume drops suddenly to near zero.",
    "Wash trading involves simultaneously buying and selling the same crypto asset to generate artificial trading volume. Detection: same wallet appearing as both sender and receiver in related transactions, identical amounts traded back and forth, trades happening at regular exact intervals.",
    "Blockchain analytics red flags for money laundering: transactions going through multiple wallets in rapid succession, use of privacy coins like Monero to obscure trail, mixing services or tumblers detected in transaction path, funds originating from darknet market wallets.",
    "FATF guidelines for crypto: exchanges must implement KYC for transactions over $1,000, report suspicious transactions, maintain records for 5 years. The travel rule requires sharing sender and receiver information for transfers over $3,000.",
    "Phishing and account takeover in crypto: sudden transaction from a wallet that has never transacted before, transaction at unusual hours, transaction amount much larger than historical average, receiver wallet created less than 24 hours ago.",
    "High risk jurisdictions for crypto transactions: transactions involving wallets linked to sanctioned countries should be automatically flagged. Cross-border transactions with high-risk jurisdictions require enhanced due diligence.",
    "Velocity checks in crypto fraud detection: more than 10 transactions in 1 hour from the same wallet is suspicious, more than 3 failed transactions followed by a successful one indicates credential stuffing, rapid movement of funds to multiple wallets within minutes indicates layering.",
    "Crypto exchange compliance: exchanges must file Suspicious Activity Reports with FinCEN for transactions over $5,000 that appear suspicious. Customer due diligence must include identity verification and source of funds verification for large transactions.",
]

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")
    return Groq(api_key=GROQ_API_KEY)


def trim_messages(messages, limit=8):
    return messages[-limit:]


def trim_history(items, limit=40):
    return items[-limit:]


def score_to_colors(fraud_score):
    if fraud_score < 30:
        return {"metric_bg": "#166534", "badge_bg": "#14532d", "font": "white"}
    if fraud_score <= 60:
        return {"metric_bg": "#ca8a04", "badge_bg": "#a16207", "font": "#111827"}
    return {"metric_bg": "#b91c1c", "badge_bg": "#991b1b", "font": "white"}


def risk_level_to_badge_color(risk_level):
    mapping = {
        "LOW": ("#166534", "white"),
        "MEDIUM": ("#ca8a04", "#111827"),
        "HIGH": ("#ea580c", "white"),
        "CRITICAL": ("#b91c1c", "white"),
    }
    return mapping.get(risk_level, ("#6b7280", "white"))


def recommendation_to_banner(rec):
    mapping = {
        "APPROVE": ("#166534", "white"),
        "REVIEW": ("#ca8a04", "#111827"),
        "BLOCK": ("#b91c1c", "white"),
    }
    return mapping.get(rec, ("#6b7280", "white"))


def extract_first_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in the model output.")
    return text[start : end + 1]


def normalize_fraud_response(obj):
    required_fields = [
        "fraud_score",
        "risk_level",
        "fraud_indicators",
        "legitimate_indicators",
        "recommendation",
        "explanation",
        "similar_fraud_patterns",
    ]
    missing = [field for field in required_fields if field not in obj]
    if missing:
        raise ValueError(f"Missing required fields in JSON: {missing}")

    fraud_score = int(float(obj["fraud_score"]))
    risk_level = str(obj["risk_level"]).upper()
    recommendation = str(obj["recommendation"]).upper()
    fraud_score = max(0, min(100, fraud_score))

    if risk_level not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise ValueError(f"Invalid risk_level: {risk_level}")
    if recommendation not in {"APPROVE", "REVIEW", "BLOCK"}:
        raise ValueError(f"Invalid recommendation: {recommendation}")

    fraud_indicators = obj["fraud_indicators"]
    legitimate_indicators = obj["legitimate_indicators"]
    similar_fraud_patterns = obj["similar_fraud_patterns"]
    explanation = str(obj["explanation"]).strip()

    if not isinstance(fraud_indicators, list) or not all(isinstance(item, str) for item in fraud_indicators):
        raise ValueError("fraud_indicators must be a list of strings.")
    if not isinstance(legitimate_indicators, list) or not all(
        isinstance(item, str) for item in legitimate_indicators
    ):
        raise ValueError("legitimate_indicators must be a list of strings.")
    if not isinstance(similar_fraud_patterns, list) or not all(
        isinstance(item, str) for item in similar_fraud_patterns
    ):
        raise ValueError("similar_fraud_patterns must be a list of strings.")
    if not explanation:
        raise ValueError("explanation must be a non-empty string.")

    return {
        "fraud_score": fraud_score,
        "risk_level": risk_level,
        "fraud_indicators": fraud_indicators,
        "legitimate_indicators": legitimate_indicators,
        "recommendation": recommendation,
        "explanation": explanation,
        "similar_fraud_patterns": similar_fraud_patterns,
    }


def analyze_transaction_with_llm(transaction_fields):
    prompt = (
        "You are a crypto fraud detection expert and compliance officer.\n"
        "Analyze the following crypto transaction and return only a valid JSON object.\n"
        "Do not include markdown, code fences, or extra text.\n\n"
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

    client = get_groq_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content or ""
    extracted = extract_first_json_object(raw)
    obj = json.loads(extracted)
    return normalize_fraud_response(obj)


def build_analysis_card(analysis):
    score = int(analysis["fraud_score"])
    colors = score_to_colors(score)
    risk_bg, risk_font = risk_level_to_badge_color(analysis["risk_level"])
    rec_bg, rec_font = recommendation_to_banner(analysis["recommendation"])
    return {
        "score": score,
        "colors": colors,
        "risk_bg": risk_bg,
        "risk_font": risk_font,
        "rec_bg": rec_bg,
        "rec_font": rec_font,
        "analysis": analysis,
    }


def generate_wallet(prefix="0x", length=40):
    chars = "0123456789abcdef"
    return prefix + "".join(random.choice(chars) for _ in range(length))


def generate_sample_transactions_csv_bytes():
    random.seed(42)
    now = datetime.now()
    rows = []
    threshold = 10000
    fraudulent_senders = [
        ("0xabc123abc123abc123abc123abc123abc123abcd", "0x9f1e2d3c4b5a6f7e8d9c0b1a2c3d4e5f6a7b8c9d"),
        ("0x1111222233334444555566667777888899990000", "0x22223333444455556666777788889999aaaabbbb"),
        ("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "0xcafebabecafebabecafebabecafebabecafebabe"),
    ]

    for index in range(10):
        tx_id = f"tx_{index + 1:04d}"
        crypto_type = random.choice(CRYPTO_TYPES)
        is_cross_border = random.random() < 0.35
        sender_wallet = generate_wallet()
        receiver_wallet = generate_wallet()
        amount = round(random.uniform(50, 45000), 2)
        if random.random() < 0.25:
            amount = round(threshold - random.uniform(10, 650), 2)

        sender_age_days = random.randint(1, 2500)
        transactions_last_24h = random.randint(0, 25)
        transaction_fee_usd = round(random.uniform(0.5, 55.0), 2)
        timestamp = (now - timedelta(hours=random.randint(1, 720))).strftime("%Y-%m-%d %H:%M:%S")

        if index in (2, 5, 8):
            sender_wallet, receiver_wallet = fraudulent_senders[(index // 3) % len(fraudulent_senders)]
            sender_age_days = random.randint(0, 45)
            transactions_last_24h = random.randint(18, 60)
            amount = round(threshold - random.uniform(1, 75), 2)
            is_cross_border = True
            transaction_fee_usd = round(random.uniform(1.0, 20.0), 2)

        if index == 7:
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

    frame = pd.DataFrame(rows)
    return frame.to_csv(index=False).encode("utf-8")


def coerce_is_cross_border(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def parse_dataframe_row(row):
    timestamp_value = row.get("timestamp")
    parsed_time = pd.to_datetime(timestamp_value, errors="coerce")
    timestamp_string = (
        parsed_time.strftime("%Y-%m-%d %H:%M:%S") if parsed_time is not pd.NaT else str(timestamp_value)
    )
    return {
        "transaction_id": str(row.get("transaction_id")),
        "sender_wallet": str(row.get("sender_wallet")),
        "receiver_wallet": str(row.get("receiver_wallet")),
        "amount_usd": float(row.get("amount_usd")),
        "crypto_type": str(row.get("crypto_type")),
        "timestamp": timestamp_string,
        "sender_account_age_days": int(float(row.get("sender_account_age_days"))),
        "transactions_last_24h": int(float(row.get("transactions_last_24h"))),
        "is_cross_border": bool(coerce_is_cross_border(row.get("is_cross_border"))),
        "transaction_fee_usd": float(row.get("transaction_fee_usd")),
    }


def summarize_record(transaction, analysis):
    return {
        "transaction_id": transaction["transaction_id"],
        "amount_usd": float(transaction["amount_usd"]),
        "risk_level": analysis["risk_level"],
        "fraud_score": int(analysis["fraud_score"]),
        "recommendation": analysis["recommendation"],
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
    }


def make_dashboard(history_items):
    if not history_items:
        return None

    frame = pd.DataFrame(history_items)
    risk_colors = {"LOW": "#166534", "MEDIUM": "#ca8a04", "HIGH": "#ea580c", "CRITICAL": "#b91c1c"}
    rec_colors = {"APPROVE": "#166534", "REVIEW": "#ca8a04", "BLOCK": "#b91c1c"}

    pie = px.pie(
        frame,
        names="risk_level",
        hole=0.35,
        color="risk_level",
        color_discrete_map=risk_colors,
    )
    pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")

    histogram = go.Figure()
    histogram.add_trace(go.Histogram(x=frame["fraud_score"], nbinsx=20, marker_color="#4338ca"))
    histogram.add_vline(x=60, line_width=3, line_dash="dash", line_color="#ef4444")
    histogram.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Fraud Score",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
    )

    scatter = px.scatter(
        frame,
        x="amount_usd",
        y="fraud_score",
        color="risk_level",
        color_discrete_map=risk_colors,
        hover_data=["transaction_id", "recommendation"],
    )
    scatter.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Amount (USD)",
        yaxis_title="Fraud Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
    )

    rec_counts = frame["recommendation"].value_counts().reindex(["APPROVE", "REVIEW", "BLOCK"]).fillna(0)
    bars = go.Figure(
        go.Bar(
            x=rec_counts.values,
            y=rec_counts.index,
            orientation="h",
            marker_color=[rec_colors.get(key, "#6b7280") for key in rec_counts.index],
        )
    )
    bars.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Count",
        yaxis_title="Recommendation",
        yaxis=dict(categoryorder="array", categoryarray=["APPROVE", "REVIEW", "BLOCK"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
    )

    total_flagged = int(frame["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
    total_blocked = int((frame["recommendation"] == "BLOCK").sum())
    total_approved = int((frame["recommendation"] == "APPROVE").sum())
    highest = frame.sort_values(by="fraud_score", ascending=False).iloc[0].to_dict()
    flagged_rows = (
        frame[frame["risk_level"].isin(["HIGH", "CRITICAL"])]
        .sort_values(by="fraud_score", ascending=False)
        .to_dict(orient="records")
    )

    figures = {
        "pie": json.dumps(pie, cls=PlotlyJSONEncoder),
        "histogram": json.dumps(histogram, cls=PlotlyJSONEncoder),
        "scatter": json.dumps(scatter, cls=PlotlyJSONEncoder),
        "bars": json.dumps(bars, cls=PlotlyJSONEncoder),
    }

    stats = {
        "total_flagged": total_flagged,
        "total_blocked": total_blocked,
        "total_approved": total_approved,
        "highest": highest,
    }
    return {"figures": figures, "stats": stats, "flagged_rows": flagged_rows}


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_knowledge(question, limit=3):
    question_tokens = tokenize(question)
    scored = []
    for index, chunk in enumerate(KNOWLEDGE_CHUNKS, start=1):
        chunk_tokens = tokenize(chunk)
        overlap = len(question_tokens & chunk_tokens)
        scored.append((overlap, index, chunk))
    ranked = [item for item in sorted(scored, reverse=True) if item[0] > 0]
    if not ranked:
        ranked = [(0, index, chunk) for index, chunk in enumerate(KNOWLEDGE_CHUNKS[:limit], start=1)]
    return ranked[:limit]


def rag_ask(question):
    retrieved = retrieve_knowledge(question, limit=3)
    context_chunks = [f"Chunk {chunk_id}: {chunk}" for _, chunk_id, chunk in retrieved]
    context = "\n\n".join(context_chunks)

    client = get_groq_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a crypto fraud detection expert and compliance officer. "
                    "Answer questions using only the provided context. If the answer is not in the context, "
                    "say that the specific information is not in the current knowledge base."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    return answer, context_chunks


def default_form_values():
    return {
        "transaction_id": "tx_0001",
        "sender_wallet": "0x1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9012",
        "receiver_wallet": "0x9f8e7d6c5b4a39281716151413121110fedcba98",
        "amount_usd": 2500.00,
        "crypto_type": "Ethereum",
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "sender_account_age_days": 120,
        "transactions_last_24h": 3,
        "is_cross_border": False,
        "transaction_fee_usd": 12.50,
    }


def build_page_context(**overrides):
    history = session.get("history", [])
    rag_messages = session.get("rag_messages", [])
    context = {
        "active_tab": "single",
        "single_form": default_form_values(),
        "single_result": None,
        "single_error": None,
        "batch_rows": [],
        "batch_error": None,
        "batch_summary": None,
        "dashboard": make_dashboard(history),
        "history_count": len(history),
        "rag_messages": rag_messages,
        "rag_error": None,
        "suggested_questions": [
            "What is structuring fraud?",
            "What are FATF guidelines?",
            "How do I detect wash trading?",
            "What are velocity check rules?",
        ],
    }
    context.update(overrides)
    return context


@app.get("/")
def home():
    return render_template("index.html", **build_page_context())


@app.get("/sample.csv")
def sample_csv():
    return Response(
        generate_sample_transactions_csv_bytes(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=shieldx_sample_transactions.csv"},
    )


@app.post("/analyze")
def analyze():
    form = request.form
    single_form = {
        "transaction_id": form.get("transaction_id", "").strip(),
        "sender_wallet": form.get("sender_wallet", "").strip(),
        "receiver_wallet": form.get("receiver_wallet", "").strip(),
        "amount_usd": float(form.get("amount_usd", 0) or 0),
        "crypto_type": form.get("crypto_type", "Ethereum"),
        "timestamp": form.get("timestamp", ""),
        "sender_account_age_days": int(form.get("sender_account_age_days", 0) or 0),
        "transactions_last_24h": int(form.get("transactions_last_24h", 0) or 0),
        "is_cross_border": form.get("is_cross_border") == "on",
        "transaction_fee_usd": float(form.get("transaction_fee_usd", 0) or 0),
    }
    transaction_fields = {
        **single_form,
        "timestamp": single_form["timestamp"].replace("T", " ") if single_form["timestamp"] else "",
    }

    try:
        analysis = analyze_transaction_with_llm(transaction_fields)
        history = session.get("history", [])
        history.append(summarize_record(transaction_fields, analysis))
        session["history"] = trim_history(history)
        return render_template(
            "index.html",
            **build_page_context(
                active_tab="single",
                single_form=single_form,
                single_result=build_analysis_card(analysis),
            ),
        )
    except Exception as exc:
        return render_template(
            "index.html",
            **build_page_context(
                active_tab="single",
                single_form=single_form,
                single_error=str(exc),
            ),
        )


@app.post("/batch-analyze")
def batch_analyze():
    uploaded = request.files.get("csv_file")
    if not uploaded or not uploaded.filename:
        return render_template(
            "index.html",
            **build_page_context(active_tab="batch", batch_error="Upload a CSV file first."),
        )

    try:
        content = uploaded.read().decode("utf-8")
        frame = pd.read_csv(io.StringIO(content))
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
        missing = [column for column in expected_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        batch_rows = []
        history = session.get("history", [])
        for _, row in frame.iterrows():
            transaction = parse_dataframe_row(row)
            analysis = analyze_transaction_with_llm(transaction)
            batch_rows.append(
                {
                    "transaction_id": transaction["transaction_id"],
                    "amount_usd": transaction["amount_usd"],
                    "risk_level": analysis["risk_level"],
                    "fraud_score": analysis["fraud_score"],
                    "recommendation": analysis["recommendation"],
                }
            )
            history.append(summarize_record(transaction, analysis))

        session["history"] = trim_history(history)

        summary_frame = pd.DataFrame(batch_rows)
        batch_summary = {
            "rows": batch_rows,
            "total_flagged": int(summary_frame["risk_level"].isin(["HIGH", "CRITICAL"]).sum()),
            "total_blocked": int((summary_frame["recommendation"] == "BLOCK").sum()),
            "total_approved": int((summary_frame["recommendation"] == "APPROVE").sum()),
            "highest": summary_frame.sort_values(by="fraud_score", ascending=False).iloc[0].to_dict(),
        }
        preview_rows = frame.head(5).fillna("").to_dict(orient="records")
        return render_template(
            "index.html",
            **build_page_context(
                active_tab="batch",
                batch_rows=preview_rows,
                batch_summary=batch_summary,
            ),
        )
    except Exception as exc:
        return render_template(
            "index.html",
            **build_page_context(active_tab="batch", batch_error=str(exc)),
        )


@app.post("/rag")
def rag_chat():
    question = request.form.get("question", "").strip()
    if not question:
        return redirect(url_for("home", _anchor="rag"))

    messages = session.get("rag_messages", [])
    try:
        answer, sources = rag_ask(question)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer, "sources": sources})
        session["rag_messages"] = trim_messages(messages)
        return render_template("index.html", **build_page_context(active_tab="rag"))
    except Exception as exc:
        session["rag_messages"] = trim_messages(messages)
        return render_template(
            "index.html",
            **build_page_context(active_tab="rag", rag_error=str(exc)),
        )


@app.post("/clear-history")
def clear_history():
    session["history"] = []
    return redirect(url_for("home", _anchor="dashboard"))


@app.post("/clear-chat")
def clear_chat():
    session["rag_messages"] = []
    return redirect(url_for("home", _anchor="rag"))


if __name__ == "__main__":
    app.run(debug=True)
