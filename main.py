import os
import warnings

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.ingest import load_and_embed_data
from src.tools import search_jobs

# Suppress warnings for a clean output
warnings.filterwarnings("ignore")

# Load API Key
load_dotenv()


def main():
    # 1. Validation: Check if API Key exists
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found. Please check your .env file.")
        return

    # 2. Check if DB exists, if not, create it
    if not os.path.exists("./chroma_db"):
        print("⚡ Database not found. Initializing...")
        load_and_embed_data()

    # Fallback option if 2.5 is rate limited
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", temperature=0, google_api_key=api_key
    )

    # 4. Setup Tools (The Hands)
    tools = [search_jobs]

    # 5. Initialize the Agent
    agent_executor = create_react_agent(llm, tools)

    # 6. User Input
    print("--------------------------------------------------")
    print("🤖 WELCOME TO THE AGENTIC CAREER ADVISOR")
    print("--------------------------------------------------")

    user_resume = input("\n📄 Paste a resume summary (or press Enter for default): ")
    if not user_resume:
        user_resume = "I am a fresh graduate with skills in Python and SQL looking for backend roles."

    print(f"\nProcessing profile: {user_resume}")
    print("\n🤖 AI Agent is thinking...\n")

    # 7. Run Agent
    query = f"""
    Act as a Career Coach.
    1. Analyze this user profile: "{user_resume}"
    2. Search for the best matching jobs using your tool.
    3. Provide a final recommendation explaining WHY the job fits.
    """

    try:
        events = agent_executor.invoke({"messages": [("user", query)]})
        final_response = events["messages"][-1].content

        print("--------------------------------------------------")
        print("💡 FINAL RECOMMENDATION:")
        print("--------------------------------------------------")
        print(final_response)
        print("--------------------------------------------------")

    except Exception as e:
        print(f"❌ An error occurred during execution: {e}")


if __name__ == "__main__":
    main()
