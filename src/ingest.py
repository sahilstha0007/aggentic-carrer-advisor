import os
import pandas as pd
# --- NEW IMPORTS (Fixes Deprecation Warnings) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
CSV_PATH = "./data/jobs.csv" 

def load_and_embed_data():
    print(f"🔄 Loading data from {CSV_PATH}...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Could not find file at: {CSV_PATH}")

    try:
        df = pd.read_csv(CSV_PATH)
        df = df.fillna("")
        print(f"   Found {len(df)} jobs in the CSV.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

    docs = []
    for index, row in df.iterrows():
        full_content = (
            f"Title: {row.get('title', '')}\n"
            f"Description: {row.get('description', '')}\n"
            f"Location: {row.get('location', '')}\n"
            f"Level: {row.get('experience_level', '')}\n"
            f"Skills: {row.get('skills', '')}"
        )
        
        doc = Document(
            page_content=full_content,
            metadata={"title": row.get('title', 'Unknown')}
        )
        docs.append(doc)

    print("🧠 Generating Embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Saving to ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("✅ Database successfully updated!")
    return vector_store

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)