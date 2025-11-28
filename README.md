# 🚀 Agentic Career Advisor

**A Capstone Project for the Google x Kaggle Gen AI Course**

The **Agentic Career Advisor** is an intelligent AI application that acts as a personal recruitment coach. Unlike standard job search engines that rely on keyword matching, this agent uses **Semantic Search (RAG)** and **Agentic Reasoning** to understand a candidate's resume, actively search a database of jobs, and provide a reasoned career recommendation.

---

## 🌟 Key Gen AI Capabilities
This project demonstrates the three core pillars of Generative AI development:

1.  **Embeddings & Vector Search (RAG):**
    *   Job descriptions are converted into vector embeddings using **HuggingFace (`all-MiniLM-L6-v2`)**.
    *   Stored locally in **ChromaDB** for semantic retrieval (matching meaning, not just words).
2.  **Function Calling (Tools):**
    *   The AI has access to a custom tool: `search_jobs(query)`.
    *   The LLM decides *when* and *how* to use this tool based on the user's input.
3.  **Agentic Workflow (LangGraph):**
    *   Powered by **LangGraph** (ReAct architecture) and **Google Gemini 2.0 Flash**.
    *   The agent maintains state, executes tools, and synthesizes the final answer.

---

## 🛠️ Tech Stack

*   **LLM:** Google Gemini 2.0 Flash
*   **Orchestration:** LangGraph & LangChain
*   **Embeddings:** Sentence-Transformers (HuggingFace)
*   **Vector Store:** ChromaDB
*   **Data Handling:** Pandas (CSV ingestion)
*   **Language:** Python 3.11

---

## 📂 Project Structure

```text
agentic-career-advisor/
├── data/
│   └── jobs.csv             # The dataset containing job roles, skills, and locations
├── src/
│   ├── ingest.py            # Logic to load CSV, create embeddings, and save to ChromaDB
│   └── tools.py             # Defines the 'search_jobs' tool for the Agent
├── main.py                  # The entry point (Agent initialization and User UI)
├── requirements.txt         # Project dependencies
└── README.md                # Documentation



## ⚡ Quick Start

### 1. Setup Environment
Requires Python 3.10 - 3.12.
```bash
git clone <your-repo-url>
cd agentic-career-advisor

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt