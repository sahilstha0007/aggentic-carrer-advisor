from langchain.tools import tool
from src.ingest import get_vector_store

@tool
def search_jobs(query: str):
    """
    Searches the job database for relevant roles based on skills or summary.
    Use this tool when you need to find job listings that match a candidate.
    """
    print(f"\n🔎 [TOOL ACTION] Searching database for: '{query}'")
    
    db = get_vector_store()
    results = db.similarity_search(query, k=3)
    
    # Format results nicely for the LLM
    output = "\n".join([f"- {doc.page_content}" for doc in results])
    return output