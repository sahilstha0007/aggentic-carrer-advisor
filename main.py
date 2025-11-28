import os
import warnings
import sys

# --- SILENCE WARNINGS (For a Clean Demo) ---
warnings.filterwarnings("ignore")
# Redirect stderr to devnull temporarily if needed, but filterwarnings usually suffices
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.ingest import load_and_embed_data
from src.tools import search_jobs

load_dotenv()

def main():
    # If no database, create it
    if not os.path.exists("./chroma_db"):
        print("⚡ Database not found. Initializing...")
        load_and_embed_data()

    # LLM (Brain) - Using the modern Gemini 2.0
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # Tools (Hands)
    tools = [search_jobs]

    # Agent (Boss)
    agent = create_react_agent(llm, tools)

    print("--------------------------------------------------")
    print("🤖 WELCOME TO AGENTIC CAREER ADVISOR")
    print("--------------------------------------------------")
    
    user_resume = input("\n📄 Paste resume summary: ")

    if not user_resume:
        user_resume = "I am a fresh graduate with skills in Python and SQL looking for backend roles."

    query = f"""
    Act as a Career Advisor.
    1. Analyze this profile: "{user_resume}".
    2. Use the 'search_jobs' tool to find matching roles.
    3. IMPORTANT: You must use the tool results to give a recommendation.
    4. Provide the BEST job recommendation with clear reasons.
    """

    print("\n🤖 Thinking...\n")

    # Run the Agent
    result = agent.invoke({"messages": [("user", query)]})

    # Get Final Answer
    final_answer = result["messages"][-1].content

    print("--------------------------------------------------")
    print("💡 FINAL RECOMMENDATION")
    print("--------------------------------------------------")
    print(final_answer)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()