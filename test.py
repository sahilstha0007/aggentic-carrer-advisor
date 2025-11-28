import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load key
load_dotenv()
key = os.getenv("GOOGLE_API_KEY")

if not key:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    exit()

print(f"🔑 Checking key: {key[:5]}...{key[-5:]}")
genai.configure(api_key=key)

print("\n📡 Connecting to Google AI Studio to list available models...\n")

try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
            available_models.append(m.name)
            
    if not available_models:
        print("\n❌ No models found! Your API key might be valid, but the 'Generative Language API' is disabled in your Google Cloud Console.")
    else:
        print(f"\n🎉 Success! You have access to {len(available_models)} models.")
        print(f"👉 Please update your main.py to use one of these names (without 'models/'), e.g., '{available_models[0].replace('models/', '')}'")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    print("This usually means your API Key is invalid or you are in a blocked region.")