⚛️ Nuclear Insight Assistant
Multi-Agent RAG System for Nuclear Science Q&A
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-CPU-green)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
---
📌 What Is This?
The Nuclear Insight Assistant is a Multi-Agent Retrieval-Augmented Generation (RAG) system that answers nuclear science questions using your own documents — with zero hallucination by design.
> Ask any nuclear science question → get a **verified, multi-level answer** grounded exclusively in your indexed documents.
---
🎯 Why This Project?
Problem	Solution
AI chatbots hallucinate nuclear facts	RAG grounds every answer in real documents
Knowledge scattered across 1000s of PDFs	FAISS semantic search finds relevant chunks instantly
Answers are too technical for students	Simplifier Agent produces student-friendly version
No verification of AI answers	Critic Agent fact-checks before showing the answer
---
🧠 Multi-Agent Pipeline
```
User Query
    ↓
🔍 Retriever Agent   — FAISS cosine search (non-LLM, local)
    ↓
🧠 Explainer Agent   — Technical answer from document context (LLM)
    ↓
📚 Simplifier Agent  — Student-friendly version (LLM)
    ↓
⚠️  Critic Agent      — Fact-check & correct (LLM)
    ↓
✅ Final Verified Answer
```
Collapse Prevention Rules
Linear flow only — no loops, no recursion
Each agent runs ONCE per query
No agent calls another agent directly
Fixed structured data passed between agents
Fallback on failure — always returns an answer
---
🏗️ Project Structure
```
nuclear_insight/
├── app.py                        # Streamlit frontend
├── pipeline.py                   # 4-agent pipeline manager
├── index_documents.py            # One-time document indexing script
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
│
├── agents/
│   ├── __init__.py
│   ├── retriever_agent.py        # FAISS vector search (non-LLM)
│   ├── llm_agents.py             # Explainer, Simplifier, Critic agents
│   └── llm_caller.py             # OpenRouter API wrapper
│
├── rag/
│   ├── __init__.py
│   └── document_processor.py    # PDF/TXT ingestion, chunking, TF-IDF+SVD, FAISS
│
└── data/
    ├── docs/                     # ← Put your PDFs/TXTs here
    └── index/                    # Auto-generated FAISS index files
```
---
⚙️ Tech Stack
Component	Technology
Frontend	Streamlit
Vector Database	FAISS-CPU
Document Embedding	TF-IDF + Truncated SVD (offline, no model download)
LLM Inference	OpenRouter API (Mistral 7B default)
PDF Parsing	pypdf
Language	Python 3.10+
---
🚀 Quick Start
1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nuclear-insight-assistant.git
cd nuclear-insight-assistant
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Set your API key
```bash
cp .env.example .env
# Open .env and add your OpenRouter API key
```
Or directly in `agents/llm_caller.py`:
```python
OPENROUTER_API_KEY = "sk-or-your-key-here"
```
Get a free API key at openrouter.ai
4. Add your documents
```
data/docs/
├── reactor_physics_handbook.pdf
├── radiation_safety_guide.pdf
└── nuclear_fuel_cycle.txt
```
5. Index documents (one time only)
```bash
python index_documents.py
```
6. Launch the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser. That's it. ✅
---
📄 Supported Document Types
Format	Support
`.pdf`	✅ (digitally created PDFs only)
`.txt`	✅
`.md`	✅
Scanned PDFs	❌ (use OCR first)
Recommended Documents
DOE Nuclear Physics Handbook Vol.1
DOE Nuclear Physics Handbook Vol.2
IAEA Reactor Physics Course Material
---
🖥️ Interface
The app shows results in 4 tabs:
Tab	Content
✅ Final Answer	Critic-verified response
🧠 Technical	Explainer Agent output
📚 Simplified	Student-friendly version
🔍 Source Chunks	Retrieved FAISS chunks with scores
---
🔧 Configuration
All config is in `agents/llm_caller.py`:
```python
OPENROUTER_API_KEY = "your-key-here"   # Your API key
MODEL = "mistralai/mistral-7b-instruct" # Free tier model
TOP_K = 6                               # Chunks retrieved per query
```
Free Models on OpenRouter
`mistralai/mistral-7b-instruct` (default)
`meta-llama/llama-3-8b-instruct`
`google/gemma-7b-it`
---
📊 Performance
Operation	Time
Index 3 documents (180 pages)	~4 seconds
FAISS retrieval	< 5ms
Full pipeline (3 LLM calls)	12–18 seconds avg
Memory usage	~300 MB
---
⚠️ Limitations
Requires internet for LLM API calls
Cannot extract text from scanned PDFs
No conversation memory between queries
TF-IDF retrieval may miss semantic synonyms
---
🔮 Future Scope
[ ] Neural embeddings (sentence-transformers)
[ ] Local LLM via Ollama (fully offline)
[ ] OCR for scanned PDFs
[ ] Streaming token-by-token responses
[ ] Conversation memory
[ ] Multi-domain support
---
📜 License
MIT License — see LICENSE
---
🙏 Acknowledgements
FAISS by Meta AI Research
Streamlit
OpenRouter
scikit-learn
IAEA and US DOE for open nuclear science publications
