import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- CONFIGURATION ---
# This creates the database in your root folder
DB_PATH = "./chroma_db"
# This looks for the CSV inside the data folder
CSV_PATH = "./data/jobs.csv" 

def load_and_embed_data():
    print(f"🔄 Loading data from {CSV_PATH}...")

    # Check if the file exists at the specific path
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Could not find file at: {CSV_PATH}\n   Make sure 'jobs.csv' is inside the 'data' folder.")

    try:
        df = pd.read_csv(CSV_PATH)
        df = df.fillna("") # Fill missing values
        print(f"   Found {len(df)} jobs in the CSV.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

    docs = []
    for index, row in df.iterrows():
        # Combine columns into a single text block for the AI to read
        # Using .get() prevents errors if a column is missing
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