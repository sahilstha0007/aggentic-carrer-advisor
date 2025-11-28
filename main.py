import os
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

    # LLM (Brain)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


    # Tools (Hands)
    tools = [search_jobs]  # <-- FIXED

    # Agent (Boss)
    agent = create_react_agent(llm, tools)

    print("\n🤖 Welcome to Agentic Career Advisor!")
    user_resume = input("\nPaste resume summary: ")

    if not user_resume:
        user_resume = "I am a fresh graduate with skills in Python and SQL looking for backend roles."

    query = f"""
    Act as a Career Advisor.
    Analyze this profile: "{user_resume}".
    Use the job search tool if needed.
    Give the BEST job recommendation with clear reasons.
    """

    print("\n🤖 Thinking...\n")

    result = agent.invoke({"messages": [("user", query)]})

    final_answer = result["messages"][-1].content

    print("\n------------------------------------")
    print("💡 FINAL RECOMMENDATION")
    print("------------------------------------")
    print(final_answer)
    print("------------------------------------")

if __name__ == "__main__":
    main()
