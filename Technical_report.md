# 🧠 Technical Report: Agentic Workflow Logic

## 1. Problem Statement
Job seekers often struggle to find roles that match their specific combination of skills, experience level, and preferences (e.g., Remote vs. On-site). Traditional keyword search often fails (e.g., searching for "Software Engineer" might miss "Python Developer").

## 2. The Solution: Agentic RAG
This project solves the problem using an **Agentic RAG (Retrieval-Augmented Generation)** approach.

### The Architecture
1.  **Data Ingestion:**
    *   We load `data/jobs.csv` containing fields like Title, Description, Skills, Location, and Experience Level.
    *   These fields are concatenated into a single "Context String."
    *   We use `HuggingFaceEmbeddings` to convert this context into numerical vectors.
    *   These vectors are stored in `ChromaDB`.

2.  **The Tool (`search_jobs`):**
    *   We define a function that takes a plain English string, converts it to a vector, and performs a Cosine Similarity search against ChromaDB.
    *   This allows the AI to find "Semantically Similar" jobs. For example, a search for "AI" will correctly retrieve "Machine Learning Engineer" even if the exact word "AI" isn't in the title.

3.  **The Agent (The Brain):**
    *   We use **Gemini 2.0 Flash** initialized with **LangGraph**.
    *   The Agent is given a system prompt: *"Act as a Career Advisor."*
    *   It is given the `search_jobs` tool.

### The Reasoning Loop (Trace)
When a user inputs: *"I want to work with data in Sydney."*

1.  **Thought:** The Agent analyzes the input. It detects a location constraint ("Sydney") and a domain ("Data").
2.  **Action:** The Agent calls `search_jobs("Data jobs in Sydney")`.
3.  **Observation:** The tool returns "Cloud Architect (Sydney)" and "Data Scientist (Bangalore)".
4.  **Synthesis:** The Agent filters the results. It sees the Data Scientist role is in Bangalore, so it discards it. It sees the Cloud Architect role is in Sydney and involves AWS/Data.
5.  **Final Answer:** *"I found a Cloud Architect role in Sydney. While it is not a pure Data Analyst role, it involves data infrastructure..."*

## 3. Challenges & Solutions
*   **Dependency Management:** Python 3.14 (Bleeding Edge) caused compatibility issues with LangChain.
    *   *Solution:* Downgraded environment to Python 3.11 for stability.
*   **Model Versioning:** `gemini-1.5-pro` was deprecated/unavailable in the specific API tier.
    *   *Solution:* Upgraded to the latest `gemini-2.0-flash`.
*   **Embeddings:** Google Embeddings required complex API handling for lists.
    *   *Solution:* Switched to local `HuggingFaceEmbeddings` which are faster and free.