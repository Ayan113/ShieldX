# 🛡️ ShieldX — AI-Powered Crypto Fraud Detection

ShieldX is an AI-powered crypto fraud detection and compliance platform 
built with Python, Streamlit, LangChain, and Groq LLM.

## Features

- **Single Transaction Analyzer** — paste any crypto transaction details 
  and get an instant fraud score, risk level, and recommendation
- **Batch CSV Analyzer** — upload a CSV of transactions and analyze all 
  of them at once with a progress tracker
- **Fraud Pattern Dashboard** — live charts showing risk distribution, 
  fraud scores, and flagged transactions
- **RAG Compliance Chatbot** — ask questions about fraud patterns and 
  regulations, answered from a built-in knowledge base using RAG

## Tech Stack

- **LLM:** Groq API (llama-3.3-70b-versatile)
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB (local)
- **RAG:** LangChain
- **UI:** Streamlit
- **Charts:** Plotly

## Setup

1. Clone the repo
```bash
   git clone https://github.com/yourusername/ShieldX.git
   cd ShieldX
```

2. Create virtual environment
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies
```bash
   pip3 install -r requirements.txt
```

4. Create a `.env` file in the root folder
```
   GROQ_API_KEY=your_groq_key_here
```

5. Run the app
```bash
   streamlit run app.py
```

6. Open http://localhost:8501 in your browser

## Getting a Free Groq API Key

1. Go to https://console.groq.com
2. Sign up free
3. Click API Keys → Create API Key
4. Paste it in your `.env` file

## Project Structure
```
ShieldX/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env               # API keys (not committed)
├── .gitignore         # Git ignore rules
└── README.md          # This file
```
